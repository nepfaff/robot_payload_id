import logging
import os
import time

from abc import abstractmethod
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np

from numpy import ndarray
from pydrake.all import (
    AugmentedLagrangianNonsmooth,
    AutoDiffXd,
    Context,
    MakeVectorVariable,
    MathematicalProgram,
    MathematicalProgramResult,
    MinimumDistanceLowerBoundConstraint,
    ModelInstanceIndex,
    MultibodyPlant,
    SnoptSolver,
)

import wandb

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
    extract_numeric_data_matrix_autodiff,
    reexpress_symbolic_data_matrix,
    remove_structurally_unidentifiable_columns,
    symbolic_to_numeric_data_matrix,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import (
    create_autodiff_plant,
    eval_expression_mat,
    eval_expression_mat_derivative,
    eval_expression_vec,
)
from robot_payload_id.utils import JointData, name_constraint

from .nevergrad_augmented_lagrangian import NevergradAugmentedLagrangian
from .nevergrad_util import NevergradLossLogger, NevergradWandbLogger
from .optimal_experiment_design_base import (
    CostFunction,
    ExcitationTrajectoryOptimizerBase,
    condition_number_and_d_optimality_cost,
    condition_number_and_e_optimality_cost,
    condition_number_cost,
)


class ExcitationTrajectoryOptimizerFourier(ExcitationTrajectoryOptimizerBase):
    """
    The abstract base class for excitation trajectory optimizers that use a Fourier
    series trajectory parameterization.
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
        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
        )
        self._num_fourier_terms = num_fourier_terms
        self._omega = omega
        self._num_timesteps = num_timesteps
        self._time_horizon = time_horizon

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimizes the trajectory parameters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The optimized trajectory
                parameters `a`, `b`, and `q0`.
        """
        raise NotImplementedError


class ExcitationTrajectoryOptimizerFourierSnopt(ExcitationTrajectoryOptimizerFourier):
    """
    The excitation trajectory optimizer that uses SNOPT to optimize the trajectory.
    This optimizer uses a Fourier series parameterization.
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


class ExcitationTrajectoryOptimizerFourierBlackBox(
    ExcitationTrajectoryOptimizerFourier
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses a Fourier series parameterization.
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
        nevergrad_method: str = "NGOpt",
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
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
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
        self._nevergrad_method = nevergrad_method
        self._use_optimization_progress_bar = use_optimization_progress_bar
        self._logging_path = logging_path

        # Create decision variables
        self._a_var = MakeVectorVariable(num_joints * num_fourier_terms, "a")
        self._b_var = MakeVectorVariable(num_joints * num_fourier_terms, "b")
        self._q0_var = MakeVectorVariable(num_joints, "q0")
        self._symbolic_vars = np.concatenate([self._a_var, self._b_var, self._q0_var])

        # Set initial guess
        # Empirically, setting any terms after the first 5 to small values results
        # in a decent initial data matrix condition number
        self._initial_guess = (
            np.random.rand(len(self._symbolic_vars)) - 0.5
            if self._num_fourier_terms < 6
            else np.concatenate(
                [
                    np.random.rand(5 * self._num_joints) - 0.5,
                    (
                        np.random.rand((self._num_fourier_terms - 5) * self._num_joints)
                        - 0.5
                    )
                    * 0.01,
                    np.random.rand(5 * self._num_joints) - 0.5,
                    (
                        np.random.rand((self._num_fourier_terms - 5) * self._num_joints)
                        - 0.5
                    )
                    * 0.01,
                    np.random.rand(self._num_joints) - 0.5,
                ]
            )
        )
        parameterization = ng.p.Array(init=self._initial_guess)
        wandb.log({"initial_guess": self._initial_guess})

        # Select optimizer
        self._optimizer = ng.optimizers.registry[self._nevergrad_method](
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
        self._optimizer.suggest(parameterization.value)
        wandb.run.summary["optimizer_name"] = self._optimizer.name

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
        return condition_number_cost(W_dataTW_data_numeric)

    def _condition_number_and_d_optimality_cost(self, var_values: np.ndarray) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
        return condition_number_and_d_optimality_cost(W_dataTW_data_numeric)

    def _condition_number_and_e_optimality_cost(self, var_values: np.ndarray) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
        return condition_number_and_e_optimality_cost(W_dataTW_data_numeric)

    def _compute_joint_positions(self, var_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _joint_limit_penalty(self, var_values: np.ndarray) -> float:
        joint_positions_numeric = self._compute_joint_positions(var_values)

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

    def _log_optimization_result(
        self, a_value: np.ndarray, b_value: np.ndarray, q0_value: np.ndarray
    ) -> None:
        if self._logging_path is not None:
            # Create accumulated minimum loss plot
            losses = self._loss_logger.load()
            if len(losses) > 0:
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

        np.save(os.path.join(wandb.run.dir, "a_value.npy"), a_value)
        np.save(os.path.join(wandb.run.dir, "b_value.npy"), b_value)
        np.save(os.path.join(wandb.run.dir, "q0_value.npy"), q0_value)
        if self._logging_path is not None:
            np.save(self._logging_path / "a_value.npy", a_value)
            np.save(self._logging_path / "b_value.npy", b_value)
            np.save(self._logging_path / "q0_value.npy", q0_value)

    def _log_base_params_mapping(self, base_param_mapping: np.ndarray) -> None:
        np.save(
            os.path.join(wandb.run.dir, "base_param_mapping.npy"),
            base_param_mapping,
        )
        if self._logging_path is not None:
            np.save(self._logging_path / "base_param_mapping.npy", base_param_mapping)

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

        # Log final loss
        final_loss = recommendation.loss
        logging.info(f"Final loss: {final_loss}")
        wandb.run.summary["final_loss"] = final_loss

        # Log optimization result
        a_value, b_value, q0_value = (
            recommendation.value[: len(self._a_var)],
            recommendation.value[
                len(self._a_var) : len(self._a_var) + len(self._b_var)
            ],
            recommendation.value[len(self._a_var) + len(self._b_var) :],
        )
        self._log_optimization_result(a_value, b_value, q0_value)

        return a_value, b_value, q0_value


class ExcitationTrajectoryOptimizerFourierBlackBoxSymbolic(
    ExcitationTrajectoryOptimizerFourierBlackBox
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses fully symbolic expressions to compute the data
    matrix. This optimizer uses a Fourier series parameterization.
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
        nevergrad_method: str = "NGOpt",
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
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
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
            nevergrad_method=nevergrad_method,
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

    def _compute_joint_positions(self, var_values: np.ndarray) -> np.ndarray:
        joint_positions_numeric = eval_expression_mat(
            self._joint_data.joint_positions, self._symbolic_vars, var_values
        )
        return joint_positions_numeric


class ExcitationTrajectoryOptimizerFourierBlackBoxSymbolicNumeric(
    ExcitationTrajectoryOptimizerFourierBlackBox
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses symbolic expressions to compute the data
    matrix but does not symbolically re-express the data matrix in terms of the
    decision variables. This optimizer uses a Fourier series parameterization.
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
        nevergrad_method: str = "NGOpt",
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
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
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
            nevergrad_method=nevergrad_method,
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

    def _compute_joint_positions(self, var_values: np.ndarray) -> np.ndarray:
        joint_positions_numeric = eval_expression_mat(
            self._joint_data.joint_positions, self._symbolic_vars, var_values
        )
        return joint_positions_numeric


class ExcitationTrajectoryOptimizerFourierBlackBoxNumeric(
    ExcitationTrajectoryOptimizerFourierBlackBox
):
    """
    The excitation trajectory optimizer that uses black box optimization to optimize
    the trajectory. This optimizer uses fully numeric computations. This is faster than
    symbolic versions when the symbolic expressions are very large (substitution slower
    than numeric re-compution). This optimizer uses a Fourier series parameterization.
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
        model_path: str,
        nevergrad_method: str = "NGOpt",
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
            model_path (str): The path to the model file (e.g. SDFormat, URDF).
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            use_optimization_progress_bar (bool): Whether to show a progress bar for the
                optimization. This might lead to a small performance hit.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written. Recording logs is a callback and hence will
                slow down the optimization.
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
            nevergrad_method=nevergrad_method,
            use_optimization_progress_bar=use_optimization_progress_bar,
            logging_path=logging_path,
        )

        self._arm_components = create_arm(
            arm_file_path=model_path, num_joints=num_joints, time_step=0.0
        )
        self._ad_plant_components = create_autodiff_plant(
            arm_components=self._arm_components
        )

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
        use_progress_bar: bool = False,
    ) -> np.ndarray:
        a, b, q0 = np.split(
            var_values, [len(self._a_var), len(self._a_var) + len(self._b_var)]
        )
        joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
            num_timesteps=self._num_timesteps,
            time_horizon=self._time_horizon,
            a=a.reshape((self._num_joints, self._num_fourier_terms), order="F"),
            b=b.reshape((self._num_joints, self._num_fourier_terms), order="F"),
            q0=q0,
            omega=self._omega,
            use_progress_bar=use_progress_bar,
        )

        # Evaluate and stack symbolic data matrix
        W_data_raw, _ = extract_numeric_data_matrix_autodiff(
            arm_components=self._ad_plant_components,
            joint_data=joint_data,
            use_progress_bar=use_progress_bar,
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
        W_data = self._compute_W_data(random_var_values, use_progress_bar=False)

        # NOTE: This might lead to running out of memory for large matrices. W_data
        # is sparse and hence it might be possible to use a sparse SVD. However,
        # this would make reconstruction difficutl.
        logging.info(
            "Computing SVD for base parameter mapping. This might take a while for "
            + "large data matrices."
        )
        svd_start = time.time()
        _, S, VT = np.linalg.svd(W_data)
        logging.info(f"SVD took {timedelta(seconds=time.time() - svd_start)}")
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

    def _compute_joint_data(self, var_values: np.ndarray) -> JointData:
        a, b, q0 = np.split(
            var_values, [len(self._a_var), len(self._a_var) + len(self._b_var)]
        )
        joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
            num_timesteps=self._num_timesteps,
            time_horizon=self._time_horizon,
            a=a.reshape((self._num_joints, self._num_fourier_terms), order="F"),
            b=b.reshape((self._num_joints, self._num_fourier_terms), order="F"),
            q0=q0,
            omega=self._omega,
            use_progress_bar=False,
        )
        return joint_data

    def _compute_joint_positions(self, var_values: np.ndarray) -> np.ndarray:
        joint_data = self._compute_joint_data(var_values)
        return joint_data.joint_positions

    def optimize(self) -> Tuple[ndarray, ndarray, ndarray]:
        self._log_base_params_mapping(self._base_param_mapping)
        return super().optimize()


class ExcitationTrajectoryOptimizerFourierBlackBoxALNumeric(
    ExcitationTrajectoryOptimizerFourierBlackBoxNumeric
):
    """
    The excitation trajectory optimizer that uses black box optimization with the
    augmented Lagrangian method to optimize the trajectory. This optimizer uses fully
    numeric computations and the Fourier series parameterization.
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
        plant_context: Context,
        robot_model_instance_idx: ModelInstanceIndex,
        max_al_iterations: int,
        budget_per_iteration: int,
        mu_initial: float,
        mu_multiplier: float,
        mu_max: float,
        model_path: str,
        nevergrad_method: str = "NGOpt",
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
            max_al_iterations (int): The maximum number of augmented Lagrangian
                iterations to run.
            budget_per_iteration (int): The number of iterations to run the optimizer
                for at each augmented Lagrangian iteration.
            mu_initial (float): The initial value of the penalty weights.
            mu_multiplier (float): We will multiply mu by mu_multiplier if the
                constraints are not satisfied.
            mu_max (float): The maximum value of the penalty weights.
            model_path (str): The path to the model file (e.g. SDFormat, URDF).
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written.
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
            budget=budget_per_iteration,
            model_path=model_path,
            nevergrad_method=nevergrad_method,
            use_optimization_progress_bar=False,
            logging_path=logging_path,
        )
        self._plant_context = plant_context
        self._mu_initial = mu_initial
        self._optimizer = None

        self._prog = MathematicalProgram()

        # Create decision variables
        self._a_var = self._prog.NewContinuousVariables(
            num_joints * num_fourier_terms, "a"
        )
        self._b_var = self._prog.NewContinuousVariables(
            num_joints * num_fourier_terms, "b"
        )
        self._q0_var = self._prog.NewContinuousVariables(num_joints, "q0")
        self._symbolic_vars = np.concatenate([self._a_var, self._b_var, self._q0_var])

        # Add cost
        self._prog.AddCost(self._cost_function_func, vars=self._symbolic_vars)

        # Add constraints
        self._add_bound_constraints()
        self._add_start_and_end_point_constraints()
        self._add_collision_constraints()

        self._ng_al = NevergradAugmentedLagrangian(
            max_al_iterations=max_al_iterations,
            budget_per_iteration=budget_per_iteration,
            mu_multiplier=mu_multiplier,
            mu_max=mu_max,
            method=self._nevergrad_method,
        )

    def _get_joint_positions(self, index: int, var_values: np.ndarray) -> np.ndarray:
        """Get the joint positions for a given joint index.

        Args:
            index (int): The index of the joint.
            var_values (np.ndarray): The decision variable values.

        Returns:
            np.ndarray: The joint positions for the given joint index.
        """
        joint_positions_numeric = self._compute_joint_positions(
            var_values
        )  # Shape (T,N)
        return joint_positions_numeric[:, index].astype(float)

    def _add_bound_constraints(self) -> None:
        """Add position, velocity, and acceleration bound constraints."""

        def no_limit(lower, upper):
            return np.isinf(lower) and np.isinf(upper)

        joint_data = self._compute_joint_data(self._symbolic_vars)

        position_lower_limits = self._plant.GetPositionLowerLimits()
        position_upper_limits = self._plant.GetPositionUpperLimits()
        velocity_lower_limits = self._plant.GetVelocityLowerLimits()
        velocity_upper_limits = self._plant.GetVelocityUpperLimits()
        acceleration_lower_limits = self._plant.GetAccelerationLowerLimits()
        acceleration_upper_limits = self._plant.GetAccelerationUpperLimits()
        for i in range(self._num_timesteps):
            for j in range(self._num_joints):
                # Position bounds
                if not no_limit(position_lower_limits[j], position_upper_limits[j]):
                    name_constraint(
                        self._prog.AddConstraint(
                            joint_data.joint_positions[i, j],
                            position_lower_limits[j],
                            position_upper_limits[j],
                        ),
                        f"positionBound_joint_{j}_time_{i}",
                    )

                # Velocity bounds
                if not no_limit(velocity_lower_limits[j], velocity_upper_limits[j]):
                    name_constraint(
                        self._prog.AddConstraint(
                            joint_data.joint_velocities[i, j],
                            velocity_lower_limits[j],
                            velocity_upper_limits[j],
                        ),
                        f"velocityBound_joint_{j}_time_{i}",
                    )

                # Acceleration bounds
                if not no_limit(
                    acceleration_lower_limits[j], acceleration_upper_limits[j]
                ):
                    name_constraint(
                        self._prog.AddConstraint(
                            joint_data.joint_accelerations[i, j],
                            acceleration_lower_limits[j],
                            acceleration_upper_limits[j],
                        ),
                        f"accelerationBound_joint_{j}_time_{i}",
                    )

    def _add_start_and_end_point_constraints(self) -> None:
        """Add constraints to start and end with zero velocities/ accelerations."""
        joint_data = self._compute_joint_data(self._symbolic_vars)
        for i in range(self._num_joints):
            name_constraint(
                self._prog.AddLinearConstraint(
                    joint_data.joint_velocities[0, i] == 0.0
                ),
                f"startVelocity_joint_{i}",
            )
            name_constraint(
                self._prog.AddLinearConstraint(
                    joint_data.joint_velocities[-1, i] == 0.0
                ),
                f"endVelocity_joint_{i}",
            )
            name_constraint(
                self._prog.AddLinearConstraint(
                    joint_data.joint_accelerations[0, i] == 0.0
                ),
                f"startAcceleration_joint_{i}",
            )
            name_constraint(
                self._prog.AddLinearConstraint(
                    joint_data.joint_accelerations[-1, i] == 0.0
                ),
                f"endAcceleration_joint_{i}",
            )

    def _add_collision_constraints(self, min_distance: float = 0.01) -> None:
        """Add collision avoidance constraints."""
        joint_positions_sym = self._compute_joint_positions(self._symbolic_vars)
        constraint = MinimumDistanceLowerBoundConstraint(
            plant=self._plant,
            bound=min_distance,
            plant_context=self._plant_context,
        )

        def collision_constraint(time_idx: int, var_values: np.ndarray) -> np.ndarray:
            joint_positions_numeric = eval_expression_vec(
                joint_positions_sym[time_idx], self._symbolic_vars, var_values
            )
            return constraint.Eval(joint_positions_numeric)

        for i in range(self._num_timesteps):
            name_constraint(
                self._prog.AddConstraint(
                    func=partial(collision_constraint, i),
                    lb=[0.0],
                    ub=[1.0],
                    vars=self._symbolic_vars,
                ),
                f"collisionAvoidance_time_{i}",
            )

    def optimize(self) -> Tuple[ndarray, ndarray, ndarray]:
        # Compute the initial Lagrange multiplier guess
        num_lambda = self._ng_al.compute_num_lambda(self._prog)
        lambda_initial = np.zeros(num_lambda)

        # Optimize
        x_val, _, _ = self._ng_al.solve(
            prog_or_al_factory=self._prog,
            x_init=self._initial_guess,
            lambda_val=lambda_initial,
            mu=self._mu_initial,
            nevergrad_set_bounds=True,
        )

        # Log optimization result
        a_value, b_value, q0_value = np.split(
            x_val, [len(self._a_var), len(self._a_var) + len(self._b_var)]
        )
        self._log_optimization_result(a_value, b_value, q0_value)
        self._log_base_params_mapping(self._base_param_mapping)

        return a_value, b_value, q0_value

    @classmethod
    def optimize_parallel(
        cls,
        num_joints: int,
        cost_function: CostFunction,
        num_fourier_terms: int,
        omega: float,
        num_timesteps: int,
        time_horizon: float,
        max_al_iterations: int,
        budget_per_iteration: int,
        mu_initial: float,
        mu_multiplier: float,
        mu_max: float,
        model_path: str,
        robot_model_instance_name: str,
        num_workers: int,
        nevergrad_method: str = "NGOpt",
        logging_path: Optional[Path] = None,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Optimizes the trajectory parameters in parallel.
        NOTE: This method deals with the optimizer construction and should be called
        as a classmethod.

        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            num_fourier_terms (int): The number of Fourier terms to use for the
                trajectory parameterization.
            omega (float): The frequency of the trajectory.
            num_timesteps (int): The number of time steps to use for the trajectory.
            time_horizon (float): The time horizon/ duration of the trajectory. The
                sampling time step is computed as time_horizon / num_timesteps.
            max_al_iterations (int): The maximum number of augmented Lagrangian
                iterations to run.
            budget_per_iteration (int): The number of iterations to run the optimizer
                for at each augmented Lagrangian iteration.
            mu_initial (float): The initial value of the penalty weights.
            mu_multiplier (float): We will multiply mu by mu_multiplier if the
                constraints are not satisfied.
            mu_max (float): The maximum value of the penalty weights.
            model_path (str): The path to the model file (e.g. SDFormat, URDF).
            robot_model_instance_name (str): The name of the robot model instance.
            num_workers (int): The number of workers to use for parallel optimization.
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written.

        Returns:
            Tuple[ndarray, ndarray, ndarray]: The optimized trajectory parameters `a`,
                `b`, and `q0`.
        """
        assert num_workers > 1, "Parallel optimization requires at least 2 workers."

        def construct_optimizer():
            arm_components = create_arm(arm_file_path=model_path, num_joints=num_joints)
            plant = arm_components.plant
            plant_context = plant.GetMyContextFromRoot(
                arm_components.diagram.CreateDefaultContext()
            )
            robot_model_instance_idx = plant.GetModelInstanceByName(
                robot_model_instance_name
            )

            return cls(
                num_joints=num_joints,
                cost_function=cost_function,
                num_fourier_terms=num_fourier_terms,
                omega=omega,
                num_timesteps=num_timesteps,
                time_horizon=time_horizon,
                plant=plant,
                plant_context=plant_context,
                robot_model_instance_idx=robot_model_instance_idx,
                max_al_iterations=max_al_iterations,
                budget_per_iteration=budget_per_iteration,
                mu_initial=mu_initial,
                mu_multiplier=mu_multiplier,
                mu_max=mu_max,
                model_path=model_path,
                nevergrad_method=nevergrad_method,
                logging_path=logging_path,
            )

        def al_factory():
            """A pickable factory for AugmentedLagrangianNonsmooth."""
            optim = construct_optimizer()
            al = AugmentedLagrangianNonsmooth(prog=optim._prog, include_x_bounds=False)
            return al

        optim = construct_optimizer()
        al = AugmentedLagrangianNonsmooth(prog=optim._prog, include_x_bounds=False)
        lambda_initial = np.zeros(al.lagrangian_size())

        x_val, _, _ = optim._ng_al.solve(
            prog_or_al_factory=al_factory,
            x_init=optim._initial_guess,
            lambda_val=lambda_initial,
            mu=mu_initial,
            nevergrad_set_bounds=True,
            num_workers=num_workers,
        )

        # Log optimization result
        a_value, b_value, q0_value = np.split(
            x_val, [len(optim._a_var), len(optim._a_var) + len(optim._b_var)]
        )
        optim._log_optimization_result(a_value, b_value, q0_value)
        optim._log_base_params_mapping(optim._base_param_mapping)

        return a_value, b_value, q0_value
