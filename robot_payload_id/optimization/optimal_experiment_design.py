import enum
import logging
import os
import time

from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np

from numpy import ndarray
from pydrake.all import (
    AutoDiffXd,
    MakeVectorVariable,
    MathematicalProgram,
    MathematicalProgramResult,
    ModelInstanceIndex,
    MultibodyPlant,
    SnoptSolver,
)

import wandb

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
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

from .nevergrad_util import NevergradLossLogger, NevergradWandbLogger


class CostFunction(enum.Enum):
    CONDITION_NUMBER = "condition_number"
    CONDITION_NUMBER_AND_D_OPTIMALITY = "condition_number_and_d_optimality"
    CONDITION_NUMBER_AND_E_OPTIMALITY = "condition_number_and_e_optimality"

    def __str__(self):
        return self.value


class ExcitationTrajectoryOptimizer(ABC):
    """
    The abstract base class for excitation trajectory optimizers.
    Uses a Fourier series trajectory parameterization.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use. NOTE: Currently only
                supports condition number.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_time_steps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
        """
        self._num_joints = num_joints
        self._num_params = num_joints * 10
        self._cost_function = cost_function
        self._num_fourier_terms = num_fourier_terms
        self._omega = omega
        self._num_timesteps = num_timesteps
        self._time_horizon = time_horizon
        self._plant = plant
        self._robot_model_instance_idx = robot_model_instance_idx

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimizes the trajectory parameters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`.
        """
        raise NotImplementedError

    def _obtain_symbolic_data_matrix_and_state_vars(
        self,
        data_matrix_dir_path: Optional[Path] = None,
        model_path: Optional[str] = None,
    ) -> Tuple[ndarray, SymJointStateVariables]:
        assert not (
            data_matrix_dir_path is None and model_path is None
        ), "Must provide either data matrix dir path or model path!"

        if data_matrix_dir_path is None:  # Compute W_sym
            arm_components = create_arm(
                arm_file_path=model_path, num_joints=self._num_joints, time_step=0.0
            )
            sym_plant_components = create_symbolic_plant(
                arm_components=arm_components, use_lumped_parameters=True
            )
            sym_state_variables = sym_plant_components.state_variables
            W_sym = extract_symbolic_data_matrix(
                symbolic_plant_components=sym_plant_components
            )
        else:  # Load W_sym
            q_var = MakeVectorVariable(self._num_joints, "q")
            q_dot_var = MakeVectorVariable(self._num_joints, "\dot{q}")
            q_ddot_var = MakeVectorVariable(self._num_joints, "\ddot{q}")
            tau_var = MakeVectorVariable(self._num_joints, "\tau")
            sym_state_variables = SymJointStateVariables(
                q=q_var, q_dot=q_dot_var, q_ddot=q_ddot_var, tau=tau_var
            )
            W_sym = load_symbolic_data_matrix(
                dir_path=data_matrix_dir_path,
                sym_state_variables=sym_state_variables,
                num_joints=self._num_joints,
                num_params=self._num_params,
            )
        return W_sym, sym_state_variables


class ExcitationTrajectoryOptimizerSnopt(ExcitationTrajectoryOptimizer):
    """
    The excitation trajectory optimizer that uses SNOPT to optimize the trajectory.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
        data_matrix_dir_path: Optional[Path] = None,
        model_path: Optional[str] = None,
        snopt_out_path: Optional[Path] = None,
        use_print_vars_callback: bool = False,
        iteration_limit: int = 10000000,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use. NOTE: Currently only
                supports condition number.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_time_steps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            data_matrix_dir_path (Path): The path to the symbolic data matrix. If None,
                then the symbolic data matrix is re-computed.
            model_path (str): The path to the model file (e.g. SDFormat, URDF). Only
                used if `data_matrix_dir_path` is None.
            snopt_out_path (Path, optional): The path to write the SNOPT output to. If
                None, then no output is written.
            use_print_vars_callback (bool, optional): Whether to print the variables at
                each iteration.
            iteration_limit (int, optional): The maximum number of iterations to run the
                optimizer for.
        """
        assert (
            cost_function == CostFunction.CONDITION_NUMBER
        ), "Only condition number cost function is supported atm!"

        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            num_fourier_terms=num_fourier_terms,
            omega=omega,
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
        )

        W_sym, sym_state_variables = self._obtain_symbolic_data_matrix_and_state_vars(
            data_matrix_dir_path=data_matrix_dir_path, model_path=model_path
        )

        # Create decision variables
        self._prog = MathematicalProgram()
        self._a_var = self._prog.NewContinuousVariables(
            num_joints * num_fourier_terms, "a"
        )
        self._b_var = self._prog.NewContinuousVariables(
            num_joints * num_fourier_terms, "b"
        )
        self._q0_var = self._prog.NewContinuousVariables(num_joints, "q0")
        self._symbolic_vars = np.concatenate([self._a_var, self._b_var, self._q0_var])

        # Express symbolic data matrix in terms of decision variables
        self._joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            a=self._a_var.reshape((num_joints, num_fourier_terms)),
            b=self._b_var.reshape((num_joints, num_fourier_terms)),
            q0=self._q0_var,
            omega=omega,
        )
        W_data_raw = reexpress_symbolic_data_matrix(
            W_sym=W_sym,
            sym_state_variables=sym_state_variables,
            joint_data=self._joint_data,
        )

        # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
        W_data = remove_structurally_unidentifiable_columns(
            W_data_raw, self._symbolic_vars
        )

        self._W_dataTW_data = W_data.T @ W_data

        self._prog.AddCost(
            self._condition_number_cost_with_gradient, vars=self._symbolic_vars
        )

        self._add_joint_limit_constraints()

        # Set solver options
        self._solver = SnoptSolver()
        snopt = self._solver.solver_id()
        if snopt_out_path is not None:
            self._prog.SetSolverOption(snopt, "Print file", str(snopt_out_path))
        self._prog.SetSolverOption(snopt, "Iterations limit", iteration_limit)
        self._symbolic_var_names = [var.get_name() for var in self._symbolic_vars]
        if use_print_vars_callback:
            self._prog.AddVisualizationCallback(
                lambda x: print(dict(zip(self._symbolic_var_names, x))),
                self._symbolic_vars,
            )

    # A-Optimality
    # NOTE: Can't symbolically compute matrix inverse for matrices bigger than 4x4
    # prog.AddCost(np.trace(W_dataTW_data_inv))

    # D-Optimality
    # NOTE: This doesn't seem to work atm as the det is 0, logdet is inf
    # prog.AddCost(-log_determinant(W_dataTW_data))

    # Can't use AddMaximizeLogDeterminantCost because it requires W_dataTW_data to be
    # polynomial to ensure convexity. We don't care about the convexity
    # prog.AddMaximizeLogDeterminantCost(W_dataTW_data)

    def _condition_number_cost_with_gradient(self, vars: np.ndarray) -> AutoDiffXd:
        # Assumes that vars are of type AutoDiffXd
        var_values = [var.value() for var in vars]
        W_dataTW_data_numeric = eval_expression_mat(
            self._W_dataTW_data, self._symbolic_vars, var_values
        )
        eigenvalues, eigenvectors = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]
        assert min_eig > 0, "Minimum eigenvalue must be greater than zero!"
        condition_number = max_eig / min_eig

        W_dataTW_data_derivatives = eval_expression_mat_derivative(
            self._W_dataTW_data, self._symbolic_vars, var_values
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

    def _add_joint_limit_constraints(self):
        """Adds joint limit constraints to the optimization problem."""
        for i in range(self._num_joints):
            joint_indices = self._plant.GetJointIndices(self._robot_model_instance_idx)
            upper_limit = self._plant.get_mutable_joint(
                joint_indices[i]
            ).position_upper_limits()[0]
            lower_limit = self._plant.get_mutable_joint(
                joint_indices[i]
            ).position_lower_limits()[0]
            for j in range(self._num_timesteps):
                self._prog.AddConstraint(
                    self._joint_data.joint_positions[j, i], lower_limit, upper_limit
                )

    def set_initial_guess(self, a: np.ndarray, b: np.ndarray, q0: np.ndarray) -> None:
        """Sets the initial guess for the trajectory parameters.

        Args:
            a (np.ndarray): The initial guess for the trajectory parameters `a` of shape
                (num_joints * num_fourier_terms,).
            b (np.ndarray): The initial guess for the trajectory parameters `b` of shape
                (num_joints * num_fourier_terms,).
            q0 (np.ndarray): The initial guess for the trajectory parameters `q0` of
                shape (num_joints,).
        """
        self._prog.SetInitialGuess(self._a_var, a)
        self._prog.SetInitialGuess(self._b_var, b)
        self._prog.SetInitialGuess(self._q0_var, q0)

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimizes the trajectory parameters using SNOPT.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`. NOTE: The final parameters are returned
                regardless of whether the optimization was successful.
        """
        logging.info("Starting optimization...")
        optimization_start = time.time()
        res: MathematicalProgramResult = self._solver.Solve(self._prog)
        logging.info(
            f"Optimization took {timedelta(seconds=time.time() - optimization_start)}"
        )
        if res.is_success():
            logging.info("Solved successfully!")
            logging.info(
                "Final param values: "
                + str(
                    dict(
                        zip(
                            self._symbolic_var_names,
                            res.GetSolution(self._symbolic_vars),
                        )
                    )
                ),
            )
        else:
            logging.warning("Failed to solve!")
            logging.info(f"MathematicalProgram:\n{self._prog}")
            logging.info(f"Solution result: {res.get_solution_result()}")
            logging.info(
                f"Infeasible constraints: {res.GetInfeasibleConstraintNames(self._prog)}"
            )
            logging.info(f"Final loss: {res.get_optimal_cost()}")
            logging.info(
                "Final param values: "
                + str(
                    dict(
                        zip(
                            self._symbolic_var_names,
                            res.GetSolution(self._symbolic_vars),
                        )
                    )
                ),
            )

        return (
            res.GetSolution(self._a_var),
            res.GetSolution(self._b_var),
            res.GetSolution(self._q0_var),
        )


class ExcitationTrajectoryOptimizerBlackBox(ExcitationTrajectoryOptimizer):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory.
    NOTE: This is a mid-level class that should not be instantiated directly.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
        budget: int,
        use_optimization_progress_bar: bool = True,
        logging_path: Optional[Path] = None,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_time_steps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            budget (int): The number of iterations to run the optimizer for.
            use_optimization_progress_bar (bool): Whether to show a progress bar for the
                optimization. This might lead to a small performance hit.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written. Recording logs is a callback and hence will
                slow down the optimization.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`.
        """
        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            num_fourier_terms=num_fourier_terms,
            omega=omega,
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
        )
        self._budget = budget
        self._use_optimization_progress_bar = use_optimization_progress_bar
        self._logging_path = logging_path

        # Create decision variables
        self._a_var = MakeVectorVariable(num_joints * num_fourier_terms, "a")
        self._b_var = MakeVectorVariable(num_joints * num_fourier_terms, "b")
        self._q0_var = MakeVectorVariable(num_joints, "q0")
        self._symbolic_vars = np.concatenate([self._a_var, self._b_var, self._q0_var])

        # Set initial guess
        parameterization = ng.p.Array(
            init=np.random.rand(len(self._symbolic_vars)) - 0.5
        )
        wandb.log({"initial_guess": parameterization.value})

        # Select optimizer
        self._optimizer = ng.optimizers.NGOpt(
            parametrization=parameterization,
            budget=budget,
        )
        # self._optimizer = ng.families.NonObjectOptimizer(
        #     method="NLOPT", random_restart=True
        # )(
        #     parametrization=parameterization,
        #     budget=budget,
        # )
        # NOTE: Cost function must be pickable for parallelization
        # self._optimizer = ng.optimizers.NGOpt(
        #     parametrization=len(self._symbolic_vars),
        #     budget=budget,
        #     num_workers=16,
        # )

        if use_optimization_progress_bar:
            self._optimizer.register_callback("tell", ng.callbacks.ProgressBar())
        if logging_path is not None:
            logging_path.mkdir(parents=True, exist_ok=True)
            self._loss_logger = NevergradLossLogger(logging_path / "losses.txt")
            self._optimizer.register_callback("tell", self._loss_logger)
        self._optimizer.register_callback("tell", NevergradWandbLogger(self._optimizer))

        cost_function_to_cost = {
            CostFunction.CONDITION_NUMBER: self._condition_number_cost,
            CostFunction.CONDITION_NUMBER_AND_D_OPTIMALITY: self._condition_number_and_d_optimality_cost,
            CostFunction.CONDITION_NUMBER_AND_E_OPTIMALITY: self._condition_number_and_e_optimality_cost,
        }
        self._cost_function_func = cost_function_to_cost[cost_function]

    def _compute_W_dataTW_data_numeric(self, var_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _condition_number_cost(self, var_values: np.ndarray) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
        eigenvalues, _ = np.linalg.eigh(W_dataTW_data_numeric)
        min_eig_idx = np.argmin(eigenvalues)
        max_eig_idx = np.argmax(eigenvalues)
        min_eig = eigenvalues[min_eig_idx]
        max_eig = eigenvalues[max_eig_idx]

        if min_eig <= 0:
            return np.inf

        condition_number = max_eig / min_eig
        wandb.log({"condition_number": condition_number})
        return condition_number

    def _condition_number_and_d_optimality_cost(self, var_values: np.ndarray) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
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
        wandb.log({"condition_number": condition_number, "d_optimality": d_optimality})
        return cost

    def _condition_number_and_e_optimality_cost(self, var_values: np.ndarray) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
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
        wandb.log({"condition_number": condition_number, "e_optimality": e_optimality})
        return cost

    def _compute_joint_positions_numeric(self, var_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _joint_limit_penalty(self, var_values: np.ndarray) -> float:
        joint_positions_numeric = self._compute_joint_positions_numeric(var_values)

        num_violations = 0
        for i in range(self._num_joints):
            joint_indices = self._plant.GetJointIndices(self._robot_model_instance_idx)
            upper_limit = self._plant.get_mutable_joint(
                joint_indices[i]
            ).position_upper_limits()[0]
            lower_limit = self._plant.get_mutable_joint(
                joint_indices[i]
            ).position_lower_limits()[0]
            num_violations += np.count_nonzero(
                (joint_positions_numeric[:, i] < lower_limit)
                | (joint_positions_numeric[:, i] > upper_limit)
            )
        wandb.log({"num_joint_limit_violations": num_violations})
        return num_violations

    def _combined_objective(self, var_values: ndarray) -> float:
        return self._cost_function_func(var_values) + 100.0 * self._joint_limit_penalty(
            var_values
        )

    def optimize(self) -> Tuple[ndarray, ndarray, ndarray]:
        # Optimize in parallel
        # with futures.ProcessPoolExecutor(max_workers=self._optimizer.num_workers) as executor:
        #     recommendation = self._optimizer.minimize(
        #         self._combined_objective, executor=executor, batch_mode=True
        #     )

        logging.info("Starting optimization...")
        optimization_start = time.time()
        recommendation = self._optimizer.minimize(self._combined_objective)
        logging.info(
            f"Optimization took {timedelta(seconds=time.time() - optimization_start)}"
        )
        final_loss = recommendation.loss
        logging.info(f"Final loss: {final_loss}")
        wandb.log({"final_loss": final_loss})
        symbolic_var_names = [var.get_name() for var in self._symbolic_vars]
        logging.info(
            f"Final param values: {dict(zip(symbolic_var_names, recommendation.value))}"
        )

        if self._logging_path is not None:
            # Create accumulated minimum loss plot
            losses = self._loss_logger.load()
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
            plt.savefig(self._logging_path / "accumulated_min_losses.png")
            wandb.log({"accumulated_min_losses": plt})

        a_value, b_value, q0_value = (
            recommendation.value[: len(self._a_var)],
            recommendation.value[
                len(self._a_var) : len(self._a_var) + len(self._b_var)
            ],
            recommendation.value[len(self._a_var) + len(self._b_var) :],
        )
        np.save(os.path.join(wandb.run.dir, "a_value.npy"), a_value)
        np.save(os.path.join(wandb.run.dir, "b_value.npy"), b_value)
        np.save(os.path.join(wandb.run.dir, "q0_value.npy"), q0_value)
        if self._logging_path is not None:
            np.save(self._logging_path / "a_value.npy", a_value)
            np.save(self._logging_path / "b_value.npy", b_value)
            np.save(self._logging_path / "q0_value.npy", q0_value)
        return a_value, b_value, q0_value


class ExcitationTrajectoryOptimizerBlackBoxSymbolic(
    ExcitationTrajectoryOptimizerBlackBox
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses fully symbolic expressions to compute the data
    matrix.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
        budget: int,
        use_optimization_progress_bar: bool = True,
        logging_path: Optional[Path] = None,
        data_matrix_dir_path: Optional[Path] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_time_steps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            budget (int): The number of iterations to run the optimizer for.
            use_optimization_progress_bar (bool): Whether to show a progress bar for the
                optimization. This might lead to a small performance hit.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written. Recording logs is a callback and hence will
                slow down the optimization.
            data_matrix_dir_path (Path): The path to the symbolic data matrix. If None,
                then the symbolic data matrix is re-computed.
            model_path (str): The path to the model file (e.g. SDFormat, URDF). Only
                used if `data_matrix_dir_path` is None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`.
        """
        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            num_fourier_terms=num_fourier_terms,
            omega=omega,
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
            budget=budget,
            use_optimization_progress_bar=use_optimization_progress_bar,
            logging_path=logging_path,
        )

        W_sym, sym_state_variables = self._obtain_symbolic_data_matrix_and_state_vars(
            data_matrix_dir_path=data_matrix_dir_path, model_path=model_path
        )

        # Compute symbolic joint data in terms of the decision variables
        self._joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            a=self._a_var.reshape((num_joints, num_fourier_terms)),
            b=self._b_var.reshape((num_joints, num_fourier_terms)),
            q0=self._q0_var,
            omega=omega,
        )

        # Express symbolic data matrix in terms of decision variables
        W_data = reexpress_symbolic_data_matrix(
            W_sym=W_sym,
            sym_state_variables=sym_state_variables,
            joint_data=self._joint_data,
        )

        # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
        W_data = remove_structurally_unidentifiable_columns(W_data, self._symbolic_vars)

        self._W_dataTW_data = W_data.T @ W_data

    def _compute_W_dataTW_data_numeric(self, var_values: np.ndarray) -> np.ndarray:
        W_dataTW_data_numeric = eval_expression_mat(
            self._W_dataTW_data, self._symbolic_vars, var_values
        )
        return W_dataTW_data_numeric

    def _compute_joint_positions_numeric(self, var_values: np.ndarray) -> np.ndarray:
        joint_positions_numeric = eval_expression_mat(
            self._joint_data.joint_positions, self._symbolic_vars, var_values
        )
        return joint_positions_numeric


class ExcitationTrajectoryOptimizerBlackBoxSymbolicNumeric(
    ExcitationTrajectoryOptimizerBlackBox
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses symbolic expressions to compute the data
    matrix but does not symbolically re-express the data matrix in terms of the
    decision variables.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
        budget: int,
        use_optimization_progress_bar: bool = True,
        logging_path: Optional[Path] = None,
        data_matrix_dir_path: Optional[Path] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_time_steps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            budget (int): The number of iterations to run the optimizer for.
            use_optimization_progress_bar (bool): Whether to show a progress bar for the
                optimization. This might lead to a small performance hit.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written. Recording logs is a callback and hence will
                slow down the optimization.
            data_matrix_dir_path (Path): The path to the symbolic data matrix. If None,
                then the symbolic data matrix is re-computed.
            model_path (str): The path to the model file (e.g. SDFormat, URDF). Only
                used if `data_matrix_dir_path` is None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`.
        """
        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            num_fourier_terms=num_fourier_terms,
            omega=omega,
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
            budget=budget,
            use_optimization_progress_bar=use_optimization_progress_bar,
            logging_path=logging_path,
        )

        (
            self._W_sym,
            self._sym_state_variables,
        ) = self._obtain_symbolic_data_matrix_and_state_vars(
            data_matrix_dir_path=data_matrix_dir_path, model_path=model_path
        )

        # Compute symbolic joint data in terms of the decision variables
        self._joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            a=self._a_var.reshape((num_joints, num_fourier_terms)),
            b=self._b_var.reshape((num_joints, num_fourier_terms)),
            q0=self._q0_var,
            omega=omega,
        )

        self._joint_symbolic_vars_expressions = np.concatenate(
            [
                self._joint_data.joint_positions,
                self._joint_data.joint_velocities,
                self._joint_data.joint_accelerations,
            ],
            axis=1,
        )  # shape (num_timesteps, num_joints * 3)

        # Remove structurally unidentifiable columns to prevent
        # SolutionResult.kUnbounded
        self._identifiable_column_mask = self._compute_identifiable_column_mask()
        logging.info(
            f"{np.sum(self._identifiable_column_mask)} of "
            + f"{len(self._identifiable_column_mask)} params are identifiable."
        )

    def _compute_W_data(
        self,
        var_values: np.ndarray,
        identifiable_column_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Evaluate symbolic joint data
        joint_symbolic_vars_values = eval_expression_mat(
            self._joint_symbolic_vars_expressions, self._symbolic_vars, var_values
        )  # shape (num_timesteps, num_joints * 3)
        q_numeric, q_dot_numeric, q_ddot_numeric = np.split(
            joint_symbolic_vars_values, 3, axis=1
        )
        joint_data_numeric = JointData(
            joint_positions=q_numeric,
            joint_velocities=q_dot_numeric,
            joint_accelerations=q_ddot_numeric,
            joint_torques=np.zeros_like(q_numeric),
            sample_times_s=self._joint_data.sample_times_s,
        )

        # Evaluate and stack symbolic data matrix
        W_data_raw, _ = symbolic_to_numeric_data_matrix(
            state_variables=self._sym_state_variables,
            joint_data=joint_data_numeric,
            W_sym=(
                self._W_sym
                if identifiable_column_mask is None
                else self._W_sym[:, identifiable_column_mask]
            ),
            use_progress_bars=False,
        )
        return W_data_raw

    def _compute_identifiable_column_mask(self) -> np.ndarray:
        random_var_values = np.random.uniform(
            low=-1, high=1, size=len(self._symbolic_vars)
        )
        W_data = self._compute_W_data(random_var_values)

        _, R = np.linalg.qr(W_data)
        identifiable = np.abs(np.diag(R)) > 1e-4
        assert (
            np.sum(identifiable) > 0
        ), "No identifiable parameters! Try increasing num traj samples."
        return identifiable

    def _compute_W_dataTW_data_numeric(self, var_values: np.ndarray) -> np.ndarray:
        W_data = self._compute_W_data(var_values, self._identifiable_column_mask)

        W_dataTW_data = W_data.T @ W_data
        return W_dataTW_data

    def _compute_joint_positions_numeric(self, var_values: np.ndarray) -> np.ndarray:
        joint_positions_numeric = eval_expression_mat(
            self._joint_data.joint_positions, self._symbolic_vars, var_values
        )
        return joint_positions_numeric


class ExcitationTrajectoryOptimizerBlackBoxNumeric(
    ExcitationTrajectoryOptimizerBlackBox
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses fully numeric computations. This is faster than
    symbolic versions when the symbolic expressions are very large (substitution slower
    than numeric re-compution).
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
        budget: int,
        use_optimization_progress_bar: bool = True,
        logging_path: Optional[Path] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_time_steps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            budget (int): The number of iterations to run the optimizer for.
            use_optimization_progress_bar (bool): Whether to show a progress bar for the
                optimization. This might lead to a small performance hit.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written. Recording logs is a callback and hence will
                slow down the optimization.
            model_path (str): The path to the model file (e.g. SDFormat, URDF). Only
                used if `data_matrix_dir_path` is None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`.
        """
        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            num_fourier_terms=num_fourier_terms,
            omega=omega,
            num_timesteps=num_timesteps,
            time_horizon=time_horizon,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
            budget=budget,
            use_optimization_progress_bar=use_optimization_progress_bar,
            logging_path=logging_path,
        )

        arm_components = create_arm(
            arm_file_path=model_path, num_joints=num_joints, time_step=0.0
        )
        self._ad_plant_components = create_autodiff_plant(arm_components=arm_components)

        self._base_param_mapping = self._compute_base_param_mapping()
        logging.info(
            f"{self._base_param_mapping.shape[1]} of "
            + f"{self._base_param_mapping.shape[0]} params are identifiable."
        )
        wandb.run.summary["num_params"] = self._base_param_mapping.shape[0]
        wandb.run.summary["num_identifiable_params"] = self._base_param_mapping.shape[1]

    def _compute_W_data(
        self,
        var_values: np.ndarray,
        base_param_mapping: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        a, b, q0 = np.split(
            var_values, [len(self._a_var), len(self._a_var) + len(self._b_var)]
        )
        joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
            num_timesteps=self._num_timesteps,
            time_horizon=self._time_horizon,
            a=a.reshape((self._num_joints, self._num_fourier_terms)),
            b=b.reshape((self._num_joints, self._num_fourier_terms)),
            q0=q0,
            omega=self._omega,
        )

        # Evaluate and stack symbolic data matrix
        W_data_raw, _ = extract_numeric_data_matrix_autodiff(
            arm_components=self._ad_plant_components,
            joint_data=joint_data,
            use_prgress_bar=False,
        )

        if base_param_mapping is None:
            W_data = W_data_raw
        else:
            # Remove structurally unidentifiable columns to prevent
            # SolutionResult.kUnbounded
            W_data = np.empty(
                (self._num_timesteps * self._num_joints, base_param_mapping.shape[1])
            )
            for i in range(self._num_timesteps):
                W_data[i * self._num_joints : (i + 1) * self._num_joints, :] = (
                    W_data_raw[i * self._num_joints : (i + 1) * self._num_joints, :]
                    @ base_param_mapping
                )

        return W_data

    def _compute_base_param_mapping(self) -> np.ndarray:
        """Computes the base parameter mapping matrix that maps the full parameters to
        the identifiable parameters."""
        random_var_values = np.random.uniform(
            low=-1, high=1, size=len(self._symbolic_vars)
        )
        W_data = self._compute_W_data(random_var_values)

        _, S, VT = np.linalg.svd(W_data)
        V = VT.T
        base_param_mapping = V[:, np.abs(S) > 1e-6]

        assert (
            base_param_mapping.shape[1] > 0
        ), "No identifiable parameters! Try increasing num traj samples."
        return base_param_mapping

    def _compute_W_dataTW_data_numeric(self, var_values) -> np.ndarray:
        W_data = self._compute_W_data(var_values, self._base_param_mapping)

        W_dataTW_data = W_data.T @ W_data
        return W_dataTW_data

    def _compute_joint_positions_numeric(self, var_values: np.ndarray) -> np.ndarray:
        # TODO: Only compute the necessary joint positions
        # NOTE: It might make more sense to compute this in 'combined_objective'
        # and then pass it to all the penalty functions that require it
        a, b, q0 = np.split(
            var_values, [len(self._a_var), len(self._a_var) + len(self._b_var)]
        )
        joint_positions_numeric = (
            compute_autodiff_joint_data_from_fourier_series_traj_params1(
                num_timesteps=self._num_timesteps,
                time_horizon=self._time_horizon,
                a=a.reshape((self._num_joints, self._num_fourier_terms)),
                b=b.reshape((self._num_joints, self._num_fourier_terms)),
                q0=q0,
                omega=self._omega,
            ).joint_positions
        )
        return joint_positions_numeric

    def optimize(self) -> Tuple[ndarray, ndarray, ndarray]:
        np.save(
            os.path.join(wandb.run.dir, "base_param_mapping.npy"),
            self._base_param_mapping,
        )
        if self._logging_path is not None:
            np.save(
                self._logging_path / "base_param_mapping.npy", self._base_param_mapping
            )
        return super().optimize()
