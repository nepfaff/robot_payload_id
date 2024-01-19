import enum
import logging
import time

from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np

from pydrake.all import (
    AutoDiffXd,
    MakeVectorVariable,
    MathematicalProgram,
    MathematicalProgramResult,
    ModelInstanceIndex,
    MultibodyPlant,
    SnoptSolver,
)

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params,
    extract_numeric_data_matrix_autodiff,
    extract_symbolic_data_matrix,
    load_symbolic_data_matrix,
    reexpress_symbolic_data_matrix,
    remove_structurally_unidentifiable_columns,
    symbolic_to_numeric_data_matrix,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import (
    create_autodiff_plant,
    create_symbolic_plant,
    eval_expression_mat,
    eval_expression_mat_derivative,
)
from robot_payload_id.utils import JointData, SymJointStateVariables

from .nevergrad_util import NevergradLossLogger


class CostFunction(enum.Enum):
    CONDITION_NUMBER = "condition_number"
    CONDITION_NUMBER_AND_D_OPTIMALITY = "condition_number_and_d_optimality"
    CONDITION_NUMBER_AND_E_OPTIMALITY = "condition_number_and_e_optimality"

    def __str__(self):
        return self.value


def optimize_traj_snopt(
    num_joints: int,
    a_init: np.ndarray,
    b_init: np.ndarray,
    q0_init: np.ndarray,
    cost_function: CostFunction,
    num_fourier_terms: int,
    omega: float,
    num_timesteps: int,
    timestep: float,
    plant: MultibodyPlant,
    robot_model_instance_idx: ModelInstanceIndex,
    data_matrix_dir_path: Optional[Path] = None,
    model_path: Optional[str] = None,
    snopt_out_path: Optional[Path] = None,
    use_print_vars_callback: bool = False,
    iteration_limit: int = 10000000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimizes the trajectory parameters using SNOPT. Uses a Fourier series
    trajectory parameterization.

    Args:
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
        plant (MultibodyPlant): The plant to use for adding constraints.
        robot_model_instance_idx (ModelInstanceIndex): The model instance index of the
            robot. Used for adding constraints.
        data_matrix_dir_path (Path): The path to the symbolic data matrix. If None, then
            the symbolic data matrix is re-computed.
        model_path (str): The path to the model file (e.g. SDFormat, URDF). Only used
            if `data_matrix_dir_path` is None.
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
    assert not (
        data_matrix_dir_path is None and model_path is None
    ), "Must provide either data matrix dir path or model path!"

    # Obtain symbolic data matrix
    if data_matrix_dir_path is None:  # Compute W_sym
        arm_components = create_arm(
            arm_file_path=model_path, num_joints=num_joints, time_step=0.0
        )
        sym_plant_components = create_symbolic_plant(
            arm_components=arm_components, use_lumped_parameters=True
        )
        sym_state_variables = sym_plant_components.state_variables
        W_sym = extract_symbolic_data_matrix(
            symbolic_plant_components=sym_plant_components
        )
    else:  # Load W_sym
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

    # Joint limit constraints
    for i in range(num_joints):
        joint_indices = plant.GetJointIndices(robot_model_instance_idx)
        upper_limit = plant.get_mutable_joint(
            joint_indices[i + 1]
        ).position_upper_limits()[0]
        lower_limit = plant.get_mutable_joint(
            joint_indices[i + 1]
        ).position_lower_limits()[0]
        for j in range(num_timesteps):
            prog.AddConstraint(
                joint_data.joint_positions[j, i], lower_limit, upper_limit
            )

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
    logging.info("Starting optimization...")
    optimization_start = time.time()
    res: MathematicalProgramResult = solver.Solve(prog)
    logging.info(
        f"Optimization took {timedelta(seconds=time.time() - optimization_start)}"
    )
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
        logging.info(
            f"Infeasible constraints: {res.GetInfeasibleConstraintNames(prog)}"
        )
        logging.info(f"Final loss: {res.get_optimal_cost()}")
        logging.info(
            "Final param values: "
            + str(dict(zip(symbolic_var_names, res.GetSolution(symbolic_vars)))),
        )

    return res.GetSolution(a), res.GetSolution(b), res.GetSolution(q0)


def optimize_traj_black_box(
    num_joints: int,
    cost_function: CostFunction,
    num_fourier_terms: int,
    omega: float,
    num_timesteps: int,
    timestep: float,
    plant: MultibodyPlant,
    robot_model_instance_idx: ModelInstanceIndex,
    budget: int,
    data_matrix_dir_path: Optional[Path] = None,
    model_path: Optional[str] = None,
    use_symbolic_computations: bool = False,
    symbolically_reexpress_data_matrix: bool = True,
    use_optimization_progress_bar: bool = True,
    logging_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimizes the trajectory parameters using black box optimization. Uses a Fourier
    series trajectory parameterization.

    Args:
        num_joints (int): The number of joints in the arm.
        cost_function (CostFunction): The cost function to use.
        num_fourier_terms (int): The number of Fourier terms to use for the trajectory
            parameterization.
        omega (float): The frequency of the trajectory.
        num_time_steps (int): The number of time steps to use for the trajectory.
        timestep (float): The time step to use for the trajectory.
        plant (MultibodyPlant): The plant to use for adding constraints.
        robot_model_instance_idx (ModelInstanceIndex): The model instance index of the
            robot. Used for adding constraints.
        budget (int): The number of iterations to run the optimizer for.
        data_matrix_dir_path (Path): The path to the symbolic data matrix. If None, then
            the symbolic data matrix is re-computed.
        model_path (str): The path to the model file (e.g. SDFormat, URDF). Only used
            if `data_matrix_dir_path` is None.
        use_symbolic_computations (bool): Whether to use symbolic computations. If False,
            then the data matrix is numerically computed from scratch at each iteration.
        symbolically_reexpress_data_matrix (bool): Whether to symbolically re-express
            the data matrix. If False, then the data matrix is numerically re-expressed
            at each iteration. Only used if `use_symbolic_computations` is True.
        use_optimization_progress_bar (bool): Whether to show a progress bar for the
            optimization. This might lead to a small performance hit.
        logging_path (Path): The path to write the optimization logs to. If None, then no
            logs are written. Recording logs is a callback and hence will slow down the
            optimization.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory parameters
            `a`, `b`, and `q0`.
    """
    assert not (
        data_matrix_dir_path is None and model_path is None
    ), "Must provide either data matrix dir path or model path!"

    # Create decision variables
    a_var = MakeVectorVariable(num_joints * num_fourier_terms, "a")
    b_var = MakeVectorVariable(num_joints * num_fourier_terms, "b")
    q0_var = MakeVectorVariable(num_joints, "q0")
    symbolic_vars = np.concatenate([a_var, b_var, q0_var])

    # TODO: Refactor into smaller functions
    if use_symbolic_computations:
        # Obtain symbolic data matrix
        if data_matrix_dir_path is None:  # Compute W_sym
            arm_components = create_arm(
                arm_file_path=model_path, num_joints=num_joints, time_step=0.0
            )
            sym_plant_components = create_symbolic_plant(
                arm_components=arm_components, use_lumped_parameters=True
            )
            sym_state_variables = sym_plant_components.state_variables
            W_sym = extract_symbolic_data_matrix(
                symbolic_plant_components=sym_plant_components
            )
        else:  # Load W_sym
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

        # Compute symbolic joint data in terms of the decision variables
        joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params(
            num_timesteps=num_timesteps,
            timestep=timestep,
            a=a_var.reshape((num_joints, num_fourier_terms)),
            b=b_var.reshape((num_joints, num_fourier_terms)),
            q0=q0_var,
            omega=omega,
        )

        if symbolically_reexpress_data_matrix:
            # Express symbolic data matrix in terms of decision variables
            W_data = reexpress_symbolic_data_matrix(
                W_sym=W_sym,
                sym_state_variables=sym_state_variables,
                joint_data=joint_data,
            )

            # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
            W_data = remove_structurally_unidentifiable_columns(W_data, symbolic_vars)

            W_dataTW_data = W_data.T @ W_data

            def compute_W_dataTW_data_numeric(var_values) -> np.ndarray:
                W_dataTW_data_numeric = eval_expression_mat(
                    W_dataTW_data, symbolic_vars, var_values
                )
                return W_dataTW_data_numeric

        else:
            joint_symbolic_vars_expressions = np.concatenate(
                [
                    joint_data.joint_positions,
                    joint_data.joint_velocities,
                    joint_data.joint_accelerations,
                ],
                axis=1,
            )  # shape (num_timesteps, num_joints * 3)

            def compute_W_data(var_values, identifiable_column_mask=None) -> np.ndarray:
                # Evaluate symbolic joint data
                joint_symbolic_vars_values = eval_expression_mat(
                    joint_symbolic_vars_expressions, symbolic_vars, var_values
                )  # shape (num_timesteps, num_joints * 3)
                q_numeric, q_dot_numeric, q_ddot_numeric = np.split(
                    joint_symbolic_vars_values, 3, axis=1
                )
                joint_data_numeric = JointData(
                    joint_positions=q_numeric,
                    joint_velocities=q_dot_numeric,
                    joint_accelerations=q_ddot_numeric,
                    joint_torques=np.zeros_like(q_numeric),
                    sample_times_s=joint_data.sample_times_s,
                )

                # Evaluate and stack symbolic data matrix
                W_data_raw, _ = symbolic_to_numeric_data_matrix(
                    state_variables=sym_state_variables,
                    joint_data=joint_data_numeric,
                    W_sym=(
                        W_sym
                        if identifiable_column_mask is None
                        else W_sym[:, identifiable_column_mask]
                    ),
                    use_progress_bars=False,
                )
                return W_data_raw

            def compute_identifiable_column_mask() -> np.ndarray:
                random_var_values = np.random.uniform(
                    low=1, high=1000, size=len(symbolic_vars)
                )
                W_data = compute_W_data(random_var_values)

                _, R = np.linalg.qr(W_data)
                identifiable = np.abs(np.diag(R)) > 1e-12
                assert (
                    np.sum(identifiable) > 0
                ), "No identifiable parameters! Try increasing num traj samples."
                return identifiable

            # Remove structurally unidentifiable columns to prevent
            # SolutionResult.kUnbounded
            identifiable_column_mask = compute_identifiable_column_mask()
            logging.info(
                f"{np.sum(identifiable_column_mask)} of {len(identifiable_column_mask)} "
                + "params are identifiable."
            )

            def compute_W_dataTW_data_numeric(var_values) -> np.ndarray:
                W_data = compute_W_data(var_values, identifiable_column_mask)

                W_dataTW_data = W_data.T @ W_data
                return W_dataTW_data

    else:
        arm_components = create_arm(
            arm_file_path=model_path, num_joints=num_joints, time_step=0.0
        )
        ad_plant_components = create_autodiff_plant(arm_components=arm_components)

        def compute_W_data(var_values, identifiable_column_mask=None) -> np.ndarray:
            a, b, q0 = np.split(var_values, [len(a_var), len(a_var) + len(b_var)])
            joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params(
                num_timesteps=num_timesteps,
                timestep=timestep,
                a=a.reshape((num_joints, num_fourier_terms)),
                b=b.reshape((num_joints, num_fourier_terms)),
                q0=q0,
                omega=omega,
            )

            # Evaluate and stack symbolic data matrix
            W_data_raw, _ = extract_numeric_data_matrix_autodiff(
                arm_components=ad_plant_components,
                joint_data=joint_data,
                use_prgress_bar=False,
            )

            # Remove structurally unidentifiable columns to prevent
            # SolutionResult.kUnbounded
            W_data = (
                W_data_raw
                if identifiable_column_mask is None
                else W_data_raw[:, identifiable_column_mask]
            )
            return W_data

        def compute_identifiable_column_mask() -> np.ndarray:
            random_var_values = np.random.uniform(
                low=1, high=1000, size=len(symbolic_vars)
            )
            W_data = compute_W_data(random_var_values)

            _, R = np.linalg.qr(W_data)
            identifiable = np.abs(np.diag(R)) > 1e-4
            assert (
                np.sum(identifiable) > 0
            ), "No identifiable parameters! Try increasing num traj samples."
            return identifiable

        identifiable_column_mask = compute_identifiable_column_mask()
        logging.info(
            f"{np.sum(identifiable_column_mask)} of {len(identifiable_column_mask)} "
            + "params are identifiable."
        )

        def compute_W_dataTW_data_numeric(var_values) -> np.ndarray:
            W_data = compute_W_data(var_values, identifiable_column_mask)

            W_dataTW_data = W_data.T @ W_data
            return W_dataTW_data

    # Define cost functions
    def condition_number_cost(var_values) -> float:
        W_dataTW_data_numeric = compute_W_dataTW_data_numeric(var_values)
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        return condition_number

    def condition_number_and_d_optimality_cost(var_values) -> float:
        W_dataTW_data_numeric = compute_W_dataTW_data_numeric(var_values)
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

    def condition_number_and_e_optimality_cost(var_values) -> float:
        W_dataTW_data_numeric = compute_W_dataTW_data_numeric(var_values)
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

    def joint_limit_penalty(var_values) -> float:
        if use_symbolic_computations:
            joint_positions_numeric = eval_expression_mat(
                joint_data.joint_positions, symbolic_vars, var_values
            )
        else:
            # TODO: Only compute the necessary joint positions
            # NOTE: It might make more sense to compute this in 'combined_objective'
            # and then pass it to all the penalty functions that require it
            a, b, q0 = np.split(var_values, [len(a_var), len(a_var) + len(b_var)])
            joint_positions_numeric = (
                compute_autodiff_joint_data_from_fourier_series_traj_params(
                    num_timesteps=num_timesteps,
                    timestep=timestep,
                    a=a.reshape((num_joints, num_fourier_terms)),
                    b=b.reshape((num_joints, num_fourier_terms)),
                    q0=q0,
                    omega=omega,
                ).joint_positions
            )

        num_violations = 0
        for i in range(num_joints):
            joint_indices = plant.GetJointIndices(robot_model_instance_idx)
            upper_limit = plant.get_mutable_joint(
                joint_indices[i]
            ).position_upper_limits()[0]
            lower_limit = plant.get_mutable_joint(
                joint_indices[i]
            ).position_lower_limits()[0]
            num_violations += np.count_nonzero(
                (joint_positions_numeric[:, i] < lower_limit)
                | (joint_positions_numeric[:, i] > upper_limit)
            )
        return num_violations

    def combined_objective(var_values) -> float:
        return cost_function_to_cost[cost_function](
            var_values
        ) + 10 * joint_limit_penalty(var_values)

    # NOTE: Cost function must be pickable for parallelization
    # optimizer = ng.optimizers.NGOpt(
    #     parametrization=len(a) + len(b) + len(q0), budget=budget, num_workers=16
    # )
    # # Optimize in parallel
    # with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
    #     recommendation = optimizer.minimize(
    #         condition_number_cost, executor=executor, batch_mode=True
    #     )

    # Solve
    optimizer = ng.optimizers.NGOpt(
        parametrization=len(a_var) + len(b_var) + len(q0_var), budget=budget
    )
    if use_optimization_progress_bar:
        optimizer.register_callback("tell", ng.callbacks.ProgressBar())
    if logging_path is not None:
        logging_path.mkdir(parents=True, exist_ok=True)
        loss_logger = NevergradLossLogger(logging_path / "losses.txt")
        optimizer.register_callback("tell", loss_logger)
    logging.info("Starting optimization...")
    optimization_start = time.time()
    recommendation = optimizer.minimize(combined_objective)
    logging.info(
        f"Optimization took {timedelta(seconds=time.time() - optimization_start)}"
    )
    logging.info(f"Final loss: {recommendation.loss}")
    symbolic_var_names = [var.get_name() for var in symbolic_vars]
    logging.info(
        f"Final param values: {dict(zip(symbolic_var_names, recommendation.value))}"
    )

    if logging_path is not None:
        # Create accumulated minimum loss plot
        losses = loss_logger.load()
        # Clip to make graph more readable
        cliped_losses = np.clip(losses, a_min=None, a_max=1e8)
        accumulated_min_losses = np.minimum.accumulate(cliped_losses)
        min_loss_first_idx = np.argmin(accumulated_min_losses)
        plt.plot(accumulated_min_losses)
        plt.axvline(x=min_loss_first_idx, color="r")
        plt.yscale("log")
        plt.title("Accumulated Minimum Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Minimum loss")
        plt.legend(["Accumulated minimum loss", "First minimum loss"])
        plt.savefig(logging_path / "accumulated_min_losses.png")
        plt.show()

    a_value, b_value, q0_value = (
        recommendation.value[: len(a_var)],
        recommendation.value[len(a_var) : len(a_var) + len(b_var)],
        recommendation.value[len(a_var) + len(b_var) :],
    )
    return a_value, b_value, q0_value
