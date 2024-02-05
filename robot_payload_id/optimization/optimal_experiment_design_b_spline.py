import logging
import os
import time

from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from numpy import ndarray
from pydrake.all import (
    AugmentedLagrangianNonsmooth,
    BsplineBasis,
    BsplineTrajectory,
    Context,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    ModelInstanceIndex,
    MultibodyPlant,
)

import wandb

from robot_payload_id.data import extract_numeric_data_matrix_autodiff
from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_autodiff_plant
from robot_payload_id.utils import JointData

from .nevergrad_augmented_lagrangian import NevergradAugmentedLagrangian
from .optimal_experiment_design_base import (
    CostFunction,
    ExcitationTrajectoryOptimizerBase,
    condition_number_and_d_optimality_cost,
    condition_number_and_e_optimality_cost,
    condition_number_cost,
)


@dataclass
class BsplineTrajectoryAttributes:
    """A data class to hold the attributes of a B-spline trajectory."""

    spline_order: int
    """The order of the B-spline basis to use."""
    control_points: ndarray
    """The control points of the B-spline trajectory of shape
    (num_joints, num_control_points)."""
    knots: ndarray
    """The knots of the B-spline basis of shape (num_knots,)."""

    @classmethod
    def from_bspline_trajectory(cls, traj: BsplineTrajectory) -> None:
        """Sets the attributes from a B-spline trajectory."""
        assert traj.start_time() == 0.0, "Trajectory must start at time 0!"
        return cls(
            spline_order=traj.basis().order(),
            control_points=traj.control_points(),
            knots=np.array(traj.basis().knots()) * traj.end_time(),
        )

    def log(self, logging_path: Optional[Path] = None) -> None:
        """Logs the B-spline trajectory attributes to wandb. If logging_path is not
        None, then the attributes are also saved to disk."""
        if wandb.run is not None:
            np.save(
                os.path.join(wandb.run.dir, "spline_order.npy"),
                np.array([self.spline_order]),
            )
            np.save(
                os.path.join(wandb.run.dir, "control_points.npy"), self.control_points
            )
            np.save(os.path.join(wandb.run.dir, "knots.npy"), self.knots)
        if logging_path is not None:
            np.save(logging_path / "spline_order.npy", np.array([self.spline_order]))
            np.save(logging_path / "control_points.npy", self.control_points)
            np.save(logging_path / "knots.npy", self.knots)


class ExcitationTrajectoryOptimizerBspline(ExcitationTrajectoryOptimizerBase):
    """
    The abstract base class for excitation trajectory optimizers that use B-spline
    trajectory parameterization.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        plant: MultibodyPlant,
        robot_model_instance_idx: ModelInstanceIndex,
        num_timesteps: int,
        num_control_points: int,
        min_trajectory_duration: float,
        max_trajectory_duration: float,
        spline_order: int = 4,
        traj_initial: Optional[BsplineTrajectory] = None,
        logging_path: Optional[Path] = None,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            plant (MultibodyPlant): The plant to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            num_timesteps (int): The number of equally spaced timesteps to sample the
                trajectory at for constructing the data matrix.
            num_control_points (int): The number of control points to use for the
                B-spline trajectory.
            min_trajectory_duration (float): The minimum duration of the trajectory.
                This must be positive.
            max_trajectory_duration (float): The maximum duration of the trajectory.
            spline_order (int): The order of the B-spline basis to use.
            traj_initial (Optional[BsplineTrajectory]): The initial guess for the
                trajectory. If None, then a zero trajectory over the interval
                [0, (min_trajectory_duration + max_trajectory_duration) / 2.0] is used.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written.
        """
        assert min_trajectory_duration > 0, "min_trajectory_duration must be positive!"
        assert (
            min_trajectory_duration < max_trajectory_duration
        ), "min_trajectory_duration must be less than max_trajectory_duration!"
        # Validate initial guess
        if traj_initial is not None:
            assert traj_initial.start_time() == 0.0, "Trajectory must start at time 0!"
            assert traj_initial.end_time() <= max_trajectory_duration
            assert traj_initial.end_time() >= min_trajectory_duration
            assert traj_initial.basis().order() == spline_order
            assert traj_initial.control_points().shape[0] == num_joints
            assert traj_initial.control_points().shape[1] == num_control_points

        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
        )
        self._num_timesteps = num_timesteps
        self._min_trajectory_duration = min_trajectory_duration
        self._max_trajectory_duration = max_trajectory_duration
        self._logging_path = logging_path

        # Create optimization problem
        self._initial_traj_guess = (
            BsplineTrajectory(
                basis=BsplineBasis(
                    order=spline_order,
                    num_basis_functions=num_control_points,
                    initial_parameter_value=0.0,
                    final_parameter_value=max_trajectory_duration,
                ),
                control_points=np.zeros((num_joints, num_control_points)),
            )
            if traj_initial is None
            else traj_initial
        )
        self._trajopt = KinematicTrajectoryOptimization(self._initial_traj_guess)
        self._prog = self._trajopt.get_mutable_prog()

        if traj_initial is not None:
            self._trajopt.SetInitialGuess(traj_initial)

    @abstractmethod
    def optimize(self) -> BsplineTrajectoryAttributes:
        """Optimizes the trajectory parameters.

        Returns:
            BsplineTrajectoryAttributes: The optimized B-spline trajectory attributes.
        """
        raise NotImplementedError


class ExcitationTrajectoryOptimizerBsplineBlackBoxALNumeric(
    ExcitationTrajectoryOptimizerBspline
):
    """
    The excitation trajectory optimizer that uses black box optimization with the
    augmented Lagrangian method to optimize the trajectory. This optimizer uses fully
    numeric computations and the B-spline parameterization.
    """

    def __init__(
        self,
        num_joints: int,
        cost_function: CostFunction,
        plant: MultibodyPlant,
        plant_context: Context,
        robot_model_instance_idx: ModelInstanceIndex,
        model_path: str,
        num_timesteps: int,
        num_control_points: int,
        min_trajectory_duration: float,
        max_trajectory_duration: float,
        max_al_iterations: int,
        budget_per_iteration: int,
        mu_initial: float,
        mu_multiplier: float,
        mu_max: float,
        nevergrad_method: str = "NGOpt",
        spline_order: int = 4,
        traj_initial: Optional[BsplineTrajectory] = None,
        logging_path: Optional[Path] = None,
    ):
        """
        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            plant (MultibodyPlant): The plant to use for adding constraints.
            plant_context (Context): The plant context to use for adding constraints.
            robot_model_instance_idx (ModelInstanceIndex): The model instance index of
                the robot. Used for adding constraints.
            model_path (str): The path to the model file (e.g. SDFormat, URDF).
            num_timesteps (int): The number of equally spaced timesteps to sample the
                trajectory at for constructing the data matrix.
            num_control_points (int): The number of control points to use for the
                B-spline trajectory.
            min_trajectory_duration (float): The minimum duration of the trajectory.
                This must be positive.
            max_trajectory_duration (float): The maximum duration of the trajectory.
            max_al_iterations (int): The maximum number of augmented Lagrangian
                iterations to run.
            budget_per_iteration (int): The number of iterations to run the optimizer
                for at each augmented Lagrangian iteration.
            mu_initial (float): The initial value of the penalty weights.
            mu_multiplier (float): We will multiply mu by mu_multiplier if the
                equality constraint is not satisfied.
            mu_max (float): The maximum value of the penalty weights.
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            spline_order (int): The order of the B-spline basis to use.
            traj_initial (Optional[BsplineTrajectory]): The initial guess for the
                trajectory. If None, then a zero trajectory over the interval
                [0, (min_trajectory_duration + max_trajectory_duration) / 2.0] is used.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written.
        """
        super().__init__(
            num_joints=num_joints,
            cost_function=cost_function,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
            num_timesteps=num_timesteps,
            num_control_points=num_control_points,
            min_trajectory_duration=min_trajectory_duration,
            max_trajectory_duration=max_trajectory_duration,
            spline_order=spline_order,
            traj_initial=traj_initial,
            logging_path=logging_path,
        )
        self._plant_context = plant_context
        self._mu_initial = mu_initial
        self._nevergrad_method = nevergrad_method

        # Select cost function
        cost_function_to_cost = {
            CostFunction.CONDITION_NUMBER: self._condition_number_cost,
            CostFunction.CONDITION_NUMBER_AND_D_OPTIMALITY: self._condition_number_and_d_optimality_cost,
            CostFunction.CONDITION_NUMBER_AND_E_OPTIMALITY: self._condition_number_and_e_optimality_cost,
        }
        self._cost_function_func = cost_function_to_cost[cost_function]

        # Set initial parameter guess
        self._initial_decision_variable_guess = np.append(
            np.array(self._initial_traj_guess.control_points()).flatten(),
            self._initial_traj_guess.end_time(),
        )

        # Create autodiff plant components
        arm_components = create_arm(
            arm_file_path=model_path, num_joints=num_joints, time_step=0.0
        )
        self._ad_plant_components = create_autodiff_plant(arm_components=arm_components)

        # Compute base parameter mapping
        self._base_param_mapping = self._compute_base_param_mapping()
        logging.info(
            f"{self._base_param_mapping.shape[1]} of "
            + f"{self._base_param_mapping.shape[0]} params are identifiable."
        )
        wandb.run.summary["num_params"] = self._base_param_mapping.shape[0]
        wandb.run.summary["num_identifiable_params"] = self._base_param_mapping.shape[1]

        # Add cost
        self._prog.AddCost(
            self._cost_function_func, vars=self._prog.decision_variables()
        )

        # Add constraints
        self._trajopt.AddDurationConstraint(
            lb=min_trajectory_duration, ub=max_trajectory_duration
        )
        self._add_bound_constraints()
        self._add_start_and_end_point_constraints()
        self._add_collision_constraints()

        # TODO: Make method configurable. This should be configurable for all Nevergrad
        # optimizers.
        self._ng_al = NevergradAugmentedLagrangian(
            max_al_iterations=max_al_iterations,
            budget_per_iteration=budget_per_iteration,
            mu_multiplier=mu_multiplier,
            mu_max=mu_max,
            method=self._nevergrad_method,
        )

    def _log_base_params_mapping(self, base_param_mapping: np.ndarray) -> None:
        np.save(
            os.path.join(wandb.run.dir, "base_param_mapping.npy"),
            base_param_mapping,
        )
        if self._logging_path is not None:
            np.save(self._logging_path / "base_param_mapping.npy", base_param_mapping)

    def _extract_bspline_trajectory_attributes(
        self, var_values: np.ndarray
    ) -> BsplineTrajectoryAttributes:
        """Extracts the B-spline trajectory attributes from the decision variable
        values."""
        control_points = np.asarray(var_values[:-1]).reshape((self._num_joints, -1))
        traj_duration = np.abs(var_values[-1])
        scaled_knots = np.array(self._trajopt.basis().knots()) * traj_duration
        assert scaled_knots[0] == 0.0, "Trajectory must start at time 0!"
        return BsplineTrajectoryAttributes(
            spline_order=self._trajopt.basis().order(),
            control_points=control_points,
            knots=scaled_knots,
        )

    def _construct_and_sample_traj(self, var_values: np.ndarray) -> JointData:
        # Construct the trajectory
        bspline_traj_attributes = self._extract_bspline_trajectory_attributes(
            var_values
        )
        traj = BsplineTrajectory(
            basis=BsplineBasis(
                bspline_traj_attributes.spline_order, bspline_traj_attributes.knots
            ),
            control_points=bspline_traj_attributes.control_points,
        )

        # Sample the trajectory
        q_numeric = np.empty((self._num_timesteps, self._num_joints))
        q_dot_numeric = np.empty((self._num_timesteps, self._num_joints))
        q_ddot_numeric = np.empty((self._num_timesteps, self._num_joints))
        sample_times_s = np.linspace(
            traj.start_time(), traj.end_time(), num=self._num_timesteps
        )
        for i, t in enumerate(sample_times_s):
            q_numeric[i] = traj.value(t).flatten()
            q_dot_numeric[i] = traj.EvalDerivative(t, derivative_order=1).flatten()
            q_ddot_numeric[i] = traj.EvalDerivative(t, derivative_order=2).flatten()

        joint_data = JointData(
            joint_positions=q_numeric,
            joint_velocities=q_dot_numeric,
            joint_accelerations=q_ddot_numeric,
            joint_torques=np.zeros_like(q_numeric),
            sample_times_s=sample_times_s,
        )
        return joint_data

    def _compute_W_data(
        self,
        var_values: np.ndarray,
        base_param_mapping: Optional[np.ndarray] = None,
        use_progress_bar: bool = False,
    ) -> np.ndarray:
        joint_data = self._construct_and_sample_traj(var_values)

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

    def _compute_W_dataTW_data_numeric(self, var_values: np.ndarray) -> np.ndarray:
        W_data = self._compute_W_data(var_values, self._base_param_mapping)

        W_dataTW_data = W_data.T @ W_data
        return W_dataTW_data

    def _compute_base_param_mapping(self) -> np.ndarray:
        """Computes the base parameter mapping matrix that maps the full parameters to
        the identifiable parameters."""
        random_var_values = np.random.uniform(
            low=-1, high=1, size=len(self._prog.decision_variables())
        )
        W_data = self._compute_W_data(random_var_values)

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

    def _condition_number_cost(self, var_values: np.ndarray) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
        return condition_number_cost(W_dataTW_data_numeric)

    def _condition_number_and_d_optimality_cost(
        self, var_values: np.ndarray, d_optimality_weight: float = 0.1
    ) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
        return condition_number_and_d_optimality_cost(
            W_dataTW_data_numeric, d_optimality_weight
        )

    def _condition_number_and_e_optimality_cost(
        self, var_values: np.ndarray, e_optimality_weight: float = 1e-3
    ) -> float:
        W_dataTW_data_numeric = self._compute_W_dataTW_data_numeric(var_values)
        return condition_number_and_e_optimality_cost(
            W_dataTW_data_numeric, e_optimality_weight
        )

    def _add_bound_constraints(self) -> None:
        """Add position, velocity, and acceleration bound constraints."""
        self._trajopt.AddPositionBounds(
            lb=self._plant.GetPositionLowerLimits(),
            ub=self._plant.GetPositionUpperLimits(),
        )
        self._trajopt.AddVelocityBounds(
            lb=self._plant.GetVelocityLowerLimits(),
            ub=self._plant.GetVelocityUpperLimits(),
        )
        self._trajopt.AddAccelerationBounds(
            lb=self._plant.GetAccelerationLowerLimits(),
            ub=self._plant.GetAccelerationUpperLimits(),
        )

    def _add_start_and_end_point_constraints(self) -> None:
        """Add constraints to start and end with zero velocities/ accelerations."""
        self._trajopt.AddPathVelocityConstraint(
            lb=np.zeros(self._num_joints), ub=np.zeros(self._num_joints), s=0
        )
        self._trajopt.AddPathVelocityConstraint(
            lb=np.zeros(self._num_joints), ub=np.zeros(self._num_joints), s=1
        )
        self._trajopt.AddPathAccelerationConstraint(
            lb=np.zeros(self._num_joints), ub=np.zeros(self._num_joints), s=0
        )
        self._trajopt.AddPathAccelerationConstraint(
            lb=np.zeros(self._num_joints), ub=np.zeros(self._num_joints), s=1
        )

    def _add_collision_constraints(self, min_distance: float = 0.01) -> None:
        """Add collision avoidance constraints."""
        constraint = MinimumDistanceLowerBoundConstraint(
            plant=self._plant,
            bound=min_distance,
            plant_context=self._plant_context,
        )
        evaluate_at_s = np.linspace(0, 1, 3 * self._max_trajectory_duration * 100)
        for s in evaluate_at_s:
            self._trajopt.AddPathPositionConstraint(constraint, s)

    def optimize(self) -> BsplineTrajectoryAttributes:
        # Compute the initial Lagrange multiplier guess
        num_lambda = self._ng_al.compute_num_lambda(self._prog)
        lambda_initial = np.zeros(num_lambda)

        # Optimize
        x_val, _, _ = self._ng_al.solve(
            prog_or_al_factory=self._prog,
            x_init=self._initial_decision_variable_guess,
            lambda_val=lambda_initial,
            mu=self._mu_initial,
            nevergrad_set_bounds=True,
        )

        bspline_traj_attributes = self._extract_bspline_trajectory_attributes(x_val)
        bspline_traj_attributes.log(self._logging_path)

        return bspline_traj_attributes

    @classmethod
    def optimize_parallel(
        cls,
        num_joints: int,
        cost_function: CostFunction,
        model_path: str,
        robot_model_instance_name: str,
        num_timesteps: int,
        num_control_points: int,
        min_trajectory_duration: float,
        max_trajectory_duration: float,
        max_al_iterations: int,
        budget_per_iteration: int,
        mu_initial: float,
        mu_multiplier: float,
        mu_max: float,
        num_workers: int,
        nevergrad_method: str = "NGOpt",
        spline_order: int = 4,
        traj_initial: Optional[BsplineTrajectory] = None,
        logging_path: Optional[Path] = None,
    ) -> BsplineTrajectoryAttributes:
        """Optimizes the trajectory parameters in parallel.
        NOTE: This method deals with the optimizer construction and should be called
        as a classmethod.

        Args:
            num_joints (int): The number of joints in the arm.
            cost_function (CostFunction): The cost function to use.
            model_path (str): The path to the model file (e.g. SDFormat, URDF).
            robot_model_instance_name (str): The name of the robot model instance in the
                plant.
            num_timesteps (int): The number of equally spaced timesteps to sample the
                trajectory at for constructing the data matrix.
            num_control_points (int): The number of control points to use for the
                B-spline trajectory.
            min_trajectory_duration (float): The minimum duration of the trajectory.
                This must be positive.
            max_trajectory_duration (float): The maximum duration of the trajectory.
            max_al_iterations (int): The maximum number of augmented Lagrangian
                iterations to run.
            budget_per_iteration (int): The number of iterations to run the optimizer
                for at each augmented Lagrangian iteration.
            mu_initial (float): The initial value of the penalty weights.
            mu_multiplier (float): We will multiply mu by mu_multiplier if the
                equality constraint is not satisfied.
            mu_max (float): The maximum value of the penalty weights.
            num_workers (int): The number of workers to use for parallel optimization.
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            spline_order (int): The order of the B-spline basis to use.
            traj_initial (Optional[BsplineTrajectory]): The initial guess for the
                trajectory. If None, then a zero trajectory over the interval
                [0, (min_trajectory_duration + max_trajectory_duration) / 2.0] is used.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written.

        Returns:
            BsplineTrajectoryAttributes: The optimized B-spline trajectory attributes.
        """
        assert num_workers > 1, "Parallel optimization requires at least 2 workers."

        def construct_optimizer():
            arm_components = create_arm(arm_file_path=model_path, num_joints=num_joints)
            plant_context = arm_components.plant.GetMyContextFromRoot(
                arm_components.diagram.CreateDefaultContext()
            )
            robot_model_instance_idx = arm_components.plant.GetModelInstanceByName(
                robot_model_instance_name
            )

            return cls(
                num_joints=num_joints,
                cost_function=cost_function,
                plant=arm_components.plant,
                plant_context=plant_context,
                robot_model_instance_idx=robot_model_instance_idx,
                model_path=model_path,
                num_timesteps=num_timesteps,
                num_control_points=num_control_points,
                min_trajectory_duration=min_trajectory_duration,
                max_trajectory_duration=max_trajectory_duration,
                max_al_iterations=max_al_iterations,
                budget_per_iteration=budget_per_iteration,
                mu_initial=mu_initial,
                mu_multiplier=mu_multiplier,
                mu_max=mu_max,
                nevergrad_method=nevergrad_method,
                spline_order=spline_order,
                traj_initial=traj_initial,
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
            x_init=optim._initial_decision_variable_guess,
            lambda_val=lambda_initial,
            mu=mu_initial,
            nevergrad_set_bounds=True,
            num_workers=num_workers,
        )

        # Log optimization result
        bspline_traj_attributes = optim._extract_bspline_trajectory_attributes(x_val)
        bspline_traj_attributes.log(optim._logging_path)
        optim._log_base_params_mapping(optim._base_param_mapping)

        return bspline_traj_attributes
