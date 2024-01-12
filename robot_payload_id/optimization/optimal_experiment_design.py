import enum
import logging

from pathlib import Path
from typing import Optional, Tuple

import nevergrad as ng
import numpy as np

from pydrake.all import (
    AutoDiffXd,
    MakeVectorVariable,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
)

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params,
    compute_autodiff_joint_data_from_simple_sinusoidal_traj_params,
    load_symbolic_data_matrix,
    reexpress_symbolic_data_matrix,
    remove_structurally_unidentifiable_columns,
)
from robot_payload_id.symbolic import (
    eval_expression_mat,
    eval_expression_mat_derivative,
)
from robot_payload_id.utils import SymJointStateVariables


class CostFunction(enum.Enum):
    CONDITION_NUMBER = "condition_number"
    CONDITION_NUMBER_AND_D_OPTIMALITY = "condition_number_and_d_optimality"
    CONDITION_NUMBER_AND_E_OPTIMALITY = "condition_number_and_e_optimality"

    def __str__(self):
        return self.value


def optimize_traj_snopt(
    data_matrix_dir_path: Path,
    num_joints: int,
    a_init: np.ndarray,
    b_init: np.ndarray,
    q0_init: np.ndarray,
    cost_function: CostFunction,
    num_fourier_terms: int,
    omega: float,
    num_timesteps: int,
    timestep: float,
    snopt_out_path: Optional[Path] = None,
    use_print_vars_callback: bool = False,
    iteration_limit: int = 10000000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimizes the trajectory parameters using SNOPT. Uses a Fourier series
    trajectory parameterization.

    Args:
        data_matrix_dir_path (Path): The path to the symbolic data matrix.
        num_joints (int): The number of joints in the arm.
        a_init (np.ndarray): The initial guess for the trajectory parameters `a`.
        b_init (np.ndarray): The initial guess for the trajectory parameters `b`.
        q0_init (np.ndarray): The initial guess for the trajectory parameters `q0`.
        cost_function (CostFunction): The cost function to use. NOTE: Currently only
            supports condition number.
        num_fourier_terms (int): The number of Fourier terms to use for the trajectory
            parameterization.
        omega (float): The frequency of the trajectory.
        num_time_steps (int): The number of time steps to use for the trajectory.
        timestep (float): The time step to use for the trajectory.
        snopt_out_path (Path, optional): The path to write the SNOPT output to. If None,
            then no output is written.
        use_print_vars_callback (bool, optional): Whether to print the variables at each
            iteration.
        iteration_limit (int, optional): The maximum number of iterations to run the
            optimizer for.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory parameters
            `a`, `b`, and `q0`. NOTE: The final parameters are returned regardless of
            whether the optimization was successful.
    """
    assert (
        cost_function == CostFunction.CONDITION_NUMBER
    ), "Only condition number cost function is supported atm!"

    # Load symbolic data matrix
    q_var = MakeVectorVariable(num_joints, "q")
    q_dot_var = MakeVectorVariable(num_joints, "\dot{q}")
    q_ddot_var = MakeVectorVariable(num_joints, "\ddot{q}")
    tau_var = MakeVectorVariable(num_joints, "\tau")
    sym_state_variables = SymJointStateVariables(
        q=q_var, q_dot=q_dot_var, q_ddot=q_ddot_var, tau=tau_var
    )
    W_sym = load_symbolic_data_matrix(
        dir_path=data_matrix_dir_path,
        sym_state_variables=sym_state_variables,
        num_joints=num_joints,
        num_params=num_joints * 10,
    )

    # Create decision variables
    prog = MathematicalProgram()
    a = prog.NewContinuousVariables(num_joints * num_fourier_terms, "a")
    b = prog.NewContinuousVariables(num_joints * num_fourier_terms, "b")
    q0 = prog.NewContinuousVariables(num_joints, "q0")
    symbolic_vars = np.concatenate([a, b, q0])

    # Express symbolic data matrix in terms of decision variables
    joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params(
        num_timesteps=num_timesteps,
        timestep=timestep,
        a=a.reshape((num_joints, num_fourier_terms)),
        b=b.reshape((num_joints, num_fourier_terms)),
        q0=q0,
        omega=omega,
    )
    W_data = reexpress_symbolic_data_matrix(
        W_sym=W_sym, sym_state_variables=sym_state_variables, joint_data=joint_data
    )

    # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
    W_data = remove_structurally_unidentifiable_columns(W_data, symbolic_vars)

    W_dataTW_data = W_data.T @ W_data

    # A-Optimality
    # NOTE: Can't symbolically compute matrix inverse for matrices bigger than 4x4
    # prog.AddCost(np.trace(W_dataTW_data_inv))

    # D-Optimality
    # NOTE: This doesn't seem to work atm as the det is 0, logdet is inf
    # prog.AddCost(-log_determinant(W_dataTW_data))

    # Can't use AddMaximizeLogDeterminantCost because it requires W_dataTW_data to be
    # polynomial to ensure convexity. We don't care about the convexity
    # prog.AddMaximizeLogDeterminantCost(W_dataTW_data)

    def condition_number_cost_with_gradient(vars):
        # Assumes that vars are of type AutoDiffXd
        var_values = [var.value() for var in vars]
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, eigenvectors = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]
        assert min_eig > 0, "Minimum eigenvalue must be greater than zero!"
        condition_number = max_eig / min_eig

        W_dataTW_data_derivatives = eval_expression_mat_derivative(
            W_dataTW_data, symbolic_vars, var_values
        )
        condition_number_derivatives = np.empty(len(vars))
        # Based on first order perturbation theory
        # Same as in https://github.com/SNURobotics/optimal-excitation/blob/master/apps/optimal_excitation/functions_KUKA/getCondNumber.m#L57-L58
        # with difference that they name their min eigenvalue max and vice versa
        for i in range(len(vars)):
            condition_number_derivatives[i] = (
                (1 / min_eig)
                * eigenvectors[max_eig_idx].T
                @ W_dataTW_data_derivatives[i]
                @ eigenvectors[max_eig_idx]
            ) - (
                (max_eig / min_eig**2)
                * eigenvectors[min_eig_idx].T
                @ W_dataTW_data_derivatives[i]
                @ eigenvectors[min_eig_idx]
            )

        return AutoDiffXd(condition_number, condition_number_derivatives)

    prog.AddCost(condition_number_cost_with_gradient, vars=symbolic_vars)

    # Set initial guess
    prog.SetInitialGuess(a, a_init)
    prog.SetInitialGuess(b, b_init)
    prog.SetInitialGuess(q0, q0_init)

    # Set solver options
    solver = SnoptSolver()
    snopt = solver.solver_id()
    if snopt_out_path is not None:
        prog.SetSolverOption(snopt, "Print file", str(snopt_out_path))
    prog.SetSolverOption(snopt, "Iterations limit", iteration_limit)
    symbolic_var_names = [var.get_name() for var in symbolic_vars]
    if use_print_vars_callback:
        prog.AddVisualizationCallback(
            lambda x: print(dict(zip(symbolic_var_names, x))), symbolic_vars
        )

    # Solve
    res: MathematicalProgramResult = solver.Solve(prog)
    if res.is_success():
        logging.info("Solved successfully!")
        logging.info(
            "Final param values: "
            + str(dict(zip(symbolic_var_names, res.GetSolution(symbolic_vars)))),
        )
    else:
        logging.warning("Failed to solve!")
        logging.info(f"MathematicalProgram:\n{prog}")
        logging.info(f"Solution result: {res.get_solution_result()}")
        logging.info(f"Final loss: {res.get_optimal_cost()}")
        logging.info(
            "Final param values: "
            + str(dict(zip(symbolic_var_names, res.GetSolution(symbolic_vars)))),
        )

    return res.GetSolution(a), res.GetSolution(b), res.GetSolution(q0)


def optimize_traj_black_box(
    data_matrix_dir_path: Path,
    num_joints: int,
    cost_function: CostFunction,
    num_fourier_terms: int,
    omega: float,
    num_timesteps: int,
    timestep: float,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimizes the trajectory parameters using black box optimization. Uses a Fourier
    series trajectory parameterization.

    Args:
        data_matrix_dir_path (Path): The path to the symbolic data matrix.
        num_joints (int): The number of joints in the arm.
        cost_function (CostFunction): The cost function to use.
        num_fourier_terms (int): The number of Fourier terms to use for the trajectory
            parameterization.
        omega (float): The frequency of the trajectory.
        num_time_steps (int): The number of time steps to use for the trajectory.
        timestep (float): The time step to use for the trajectory.
        budget (int): The number of iterations to run the optimizer for.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory parameters
            `a`, `b`, and `q0`.
    """
    # Load symbolic data matrix
    q_var = MakeVectorVariable(num_joints, "q")
    q_dot_var = MakeVectorVariable(num_joints, "\dot{q}")
    q_ddot_var = MakeVectorVariable(num_joints, "\ddot{q}")
    tau_var = MakeVectorVariable(num_joints, "\tau")
    sym_state_variables = SymJointStateVariables(
        q=q_var, q_dot=q_dot_var, q_ddot=q_ddot_var, tau=tau_var
    )
    W_sym = load_symbolic_data_matrix(
        dir_path=data_matrix_dir_path,
        sym_state_variables=sym_state_variables,
        num_joints=num_joints,
        num_params=num_joints * 10,
    )

    # Create decision variables
    a = MakeVectorVariable(num_joints * num_fourier_terms, "a")
    b = MakeVectorVariable(num_joints * num_fourier_terms, "b")
    q0 = MakeVectorVariable(num_joints, "q0")
    symbolic_vars = np.concatenate([a, b, q0])

    # Express symbolic data matrix in terms of decision variables
    joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params(
        num_timesteps=num_timesteps,
        timestep=timestep,
        a=a.reshape((num_joints, num_fourier_terms)),
        b=b.reshape((num_joints, num_fourier_terms)),
        q0=q0,
        omega=omega,
    )
    W_data = reexpress_symbolic_data_matrix(
        W_sym=W_sym, sym_state_variables=sym_state_variables, joint_data=joint_data
    )

    # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
    W_data = remove_structurally_unidentifiable_columns(W_data, symbolic_vars)

    W_dataTW_data = W_data.T @ W_data

    # Define cost functions
    def condition_number_cost(var_values):
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        return condition_number

    def condition_number_and_d_optimality_cost(var_values):
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        log_det = np.log(np.prod(eigenvalues))
        d_optimality = -log_det

        d_optimality_weight = 1e-1
        cost = condition_number + d_optimality_weight * d_optimality
        return cost

    def condition_number_and_e_optimality_cost(var_values):
        W_dataTW_data_numeric = eval_expression_mat(
            W_dataTW_data, symbolic_vars, var_values
        )
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        e_optimality = -min_eig

        e_optimality_weight = 1e-3
        cost = condition_number + e_optimality_weight * e_optimality
        return cost

    cost_function_to_cost = {
        CostFunction.CONDITION_NUMBER: condition_number_cost,
        CostFunction.CONDITION_NUMBER_AND_D_OPTIMALITY: condition_number_and_d_optimality_cost,
        CostFunction.CONDITION_NUMBER_AND_E_OPTIMALITY: condition_number_and_e_optimality_cost,
    }

    # NOTE: Cost function must be pickable for parallelization
    # optimizer = ng.optimizers.NGOpt(parametrization=2, budget=500000, num_workers=16)
    # # Optimize in parallel
    # with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
    #     recommendation = optimizer.minimize(
    #         condition_number_cost, executor=executor, batch_mode=True
    #     )

    # Solve
    optimizer = ng.optimizers.NGOpt(
        parametrization=len(a) + len(b) + len(q0), budget=budget
    )
    recommendation = optimizer.minimize(cost_function_to_cost[cost_function])
    logging.info(f"Final loss: {recommendation.loss}")
    symbolic_var_names = [var.get_name() for var in symbolic_vars]
    logging.info(
        f"Final param values: {dict(zip(symbolic_var_names, recommendation.value))}"
    )

    a_value, b_value, q0_value = (
        recommendation.value[: len(a)],
        recommendation.value[len(a) : len(a) + len(b)],
        recommendation.value[len(a) + len(b) :],
    )
    return a_value, b_value, q0_value
