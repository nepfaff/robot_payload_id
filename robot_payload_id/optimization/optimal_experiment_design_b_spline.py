import logging
import os
import time

from abc import abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from pydrake.all import (
    AugmentedLagrangianNonsmooth,
    BsplineBasis,
    BsplineTrajectory,
    Context,
    KinematicTrajectoryOptimization,
    LinearEqualityConstraint,
    MinimumDistanceLowerBoundConstraint,
    ModelInstanceIndex,
    MultibodyPlant,
)

import wandb

from robot_payload_id.data import (
    compute_base_param_mapping,
    extract_numeric_data_matrix_autodiff,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_autodiff_plant
from robot_payload_id.utils import (
    BsplineTrajectoryAttributes,
    JointData,
    name_unnamed_constraints,
)

from .nevergrad_augmented_lagrangian import NevergradAugmentedLagrangian
from .optimal_experiment_design_base import (
    CostFunction,
    ExcitationTrajectoryOptimizerBase,
    condition_number_and_d_optimality_cost,
    condition_number_and_e_optimality_cost,
    condition_number_cost,
)


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
        traj_initial: Optional[Union[BsplineTrajectory, Path]] = None,
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
            traj_initial (Optional[Union[BsplineTrajectory, Path]]): The initial guess
                for the trajectory. If None, then a zero trajectory over the interval
                [0, (min_trajectory_duration + max_trajectory_duration) / 2.0] is used.
                If a path is provided, then the trajectory is loaded from the path.
                Such a path must be a directory containing 'spline_order.npy',
                'control_points.npy', and 'knots.npy'.
            logging_path (Path): The path to write the optimization logs to. If None,
                then no logs are written.
        """
        assert min_trajectory_duration > 0, "min_trajectory_duration must be positive!"
        assert (
            min_trajectory_duration <= max_trajectory_duration
        ), "min_trajectory_duration must be less than or equal to max_trajectory_duration!"
        # Validate initial guess
        if traj_initial is not None:
            if isinstance(traj_initial, Path):
                traj_attrs_initial = BsplineTrajectoryAttributes.load(traj_initial)
                traj_initial = BsplineTrajectory(
                    basis=BsplineBasis(
                        order=traj_attrs_initial.spline_order,
                        knots=traj_attrs_initial.knots,
                    ),
                    control_points=traj_attrs_initial.control_points,
                )
            assert traj_initial.start_time() == 0.0, "Trajectory must start at time 0!"
            assert traj_initial.end_time() <= max_trajectory_duration
            assert traj_initial.end_time() >= min_trajectory_duration
            assert traj_initial.basis().order() == spline_order
            assert (
                np.array(traj_initial.control_points()).shape[0] == num_control_points
            )
            assert np.array(traj_initial.control_points()).shape[1] == num_joints
            logging.info("Using initial trajectory guess.")

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
        self._num_control_points = num_control_points

        # Create optimization problem
        initial_traj_guess = (
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
        self._trajopt = KinematicTrajectoryOptimization(initial_traj_guess)
        self._prog = self._trajopt.get_mutable_prog()
        wandb.run.summary["num_decision_variables"] = self._prog.num_vars()

        self._initial_decision_variable_guess = self._prog.GetInitialGuess(
            self._prog.decision_variables()
        )

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
        add_rotor_inertia: bool,
        constraint_acceleration_endpoints: bool = False,
        nevergrad_method: str = "NGOpt",
        spline_order: int = 4,
        traj_initial: Optional[Union[BsplineTrajectory, Path]] = None,
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
            add_rotor_inertia (bool): Whether to consider rotor inertia in the dynamics.
            constraint_acceleration_endpoints (bool): Whether to add acceleration
                constraints at the start and end of the trajectory.
            nevergrad_method (str): The method to use for the Nevergrad optimizer.
                Refer to https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            spline_order (int): The order of the B-spline basis to use.
            traj_initial (Optional[Union[BsplineTrajectory, Path]]): The initial guess
                for the trajectory. If None, then a zero trajectory over the interval
                [0, (min_trajectory_duration + max_trajectory_duration) / 2.0] is used.
                If a path is provided, then the trajectory is loaded from the path.
                Such a path must be a directory containing 'spline_order.npy',
                'control_points.npy', and 'knots.npy'.
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
        self._add_rotor_inertia = add_rotor_inertia
        self._constraint_acceleration_endpoints = constraint_acceleration_endpoints
        self._nevergrad_method = nevergrad_method

        # Select cost function
        cost_function_to_cost = {
            CostFunction.CONDITION_NUMBER: self._condition_number_cost,
            CostFunction.CONDITION_NUMBER_AND_D_OPTIMALITY: self._condition_number_and_d_optimality_cost,
            CostFunction.CONDITION_NUMBER_AND_E_OPTIMALITY: self._condition_number_and_e_optimality_cost,
        }
        self._cost_function_func = cost_function_to_cost[cost_function]

        # Create autodiff plant components
        arm_components = create_arm(
            arm_file_path=model_path, num_joints=num_joints, time_step=0.0
        )
        self._ad_plant_components = create_autodiff_plant(
            arm_components=arm_components, add_rotor_inertia=add_rotor_inertia
        )

        # Compute base parameter mapping
        self._base_param_mapping = self._compute_base_param_mapping()
        logging.info(
            f"{self._base_param_mapping.shape[1]} of "
            + f"{self._base_param_mapping.shape[0]} params are identifiable."
        )
        wandb.run.summary["num_params"] = self._base_param_mapping.shape[0]
        wandb.run.summary["num_identifiable_params"] = self._base_param_mapping.shape[1]
        self._log_base_params_mapping(self._base_param_mapping)

        # Add cost
        self._prog.AddCost(
            self._cost_function_func, vars=self._prog.decision_variables()
        )

        # Add constraints
        self._add_traj_duration_constraint()
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

    def _log_base_params_mapping(self, base_param_mapping: np.ndarray) -> None:
        if wandb.run is not None:
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
        control_points = (
            np.array(var_values[:-1])
            .reshape((self._num_control_points, self._num_joints))
            .T
        )
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
            add_rotor_inertia=self._add_rotor_inertia,
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
        base_param_mapping = compute_base_param_mapping(W_data)

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

    def _add_traj_duration_constraint(self) -> None:
        if self._min_trajectory_duration == self._max_trajectory_duration:
            self._prog.AddLinearEqualityConstraint(
                self._prog.decision_variables()[-1] == self._max_trajectory_duration
            )
        else:
            self._trajopt.AddDurationConstraint(
                lb=self._min_trajectory_duration, ub=self._max_trajectory_duration
            )
        name_unnamed_constraints(self._prog, "duration")

    def _add_bound_constraints(self) -> None:
        """Add position, velocity, and acceleration bound constraints."""
        self._trajopt.AddPositionBounds(
            lb=self._plant.GetPositionLowerLimits(),
            ub=self._plant.GetPositionUpperLimits(),
        )
        name_unnamed_constraints(self._prog, "positionBounds")
        self._trajopt.AddVelocityBounds(
            lb=self._plant.GetVelocityLowerLimits(),
            ub=self._plant.GetVelocityUpperLimits(),
        )
        name_unnamed_constraints(self._prog, "velocityBounds")
        self._trajopt.AddAccelerationBounds(
            lb=self._plant.GetAccelerationLowerLimits(),
            ub=self._plant.GetAccelerationUpperLimits(),
        )
        name_unnamed_constraints(self._prog, "accelerationBounds")

    def _add_start_and_end_point_constraints(self) -> None:
        """Add constraints to start and end with zero velocities/ accelerations."""
        zero_constraint = LinearEqualityConstraint(
            Aeq=np.concatenate(
                (
                    np.zeros((self._num_joints, self._num_joints)),
                    np.eye(self._num_joints),
                ),
                axis=1,
            ),
            beq=np.zeros(self._num_joints),
        )

        self._trajopt.AddVelocityConstraintAtNormalizedTime(
            constraint=zero_constraint, s=0
        )
        name_unnamed_constraints(self._prog, "startVelocity")
        self._trajopt.AddVelocityConstraintAtNormalizedTime(
            constraint=zero_constraint, s=1
        )
        name_unnamed_constraints(self._prog, "endVelocity")

        if self._constraint_acceleration_endpoints:
            self._trajopt.AddPathAccelerationConstraint(
                lb=np.zeros(self._num_joints), ub=np.zeros(self._num_joints), s=0
            )
            name_unnamed_constraints(self._prog, "startAcceleration")
            self._trajopt.AddPathAccelerationConstraint(
                lb=np.zeros(self._num_joints), ub=np.zeros(self._num_joints), s=1
            )
            name_unnamed_constraints(self._prog, "endAcceleration")

    def _add_collision_constraints(self, min_distance: float = 0.01) -> None:
        """Add collision avoidance constraints."""
        constraint = MinimumDistanceLowerBoundConstraint(
            plant=self._plant,
            bound=min_distance,
            plant_context=self._plant_context,
        )
        evaluate_at_s = np.linspace(0, 1, int(3 * self._max_trajectory_duration * 100))
        for s in evaluate_at_s:
            self._trajopt.AddPathPositionConstraint(constraint, s)
            name_unnamed_constraints(
                self._prog, f"collisionAvoidance_normalized_time_{s}"
            )

    def _extract_and_log_optimization_result(
        self,
        var_values: np.ndarray,
        al_idx: int,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Augmented Lagrangian callback to extract and log the optimization result at
        each augmented Lagrangian iteration."""
        subpath = f"al_{al_idx}"
        if self._logging_path is not None:
            logging_path = self._logging_path / subpath
            logging_path.mkdir(exist_ok=True)

            yaml_path = logging_path / "meta_data.yaml"
            with open(yaml_path, "w") as file:
                yaml.dump(meta_data, file)
        else:
            logging_path = None

        bspline_traj_attributes = self._extract_bspline_trajectory_attributes(
            var_values
        )
        bspline_traj_attributes.log(logging_path)

        logging.info(f"Logged results for augmented Lagrangian iteration {al_idx}.")

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
            log_check_point_callback=self._extract_and_log_optimization_result,
        )

        bspline_traj_attributes = self._extract_bspline_trajectory_attributes(x_val)
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
        add_rotor_inertia: bool,
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
            add_rotor_inertia (bool): Whether to consider rotor inertia in the dynamics.
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
                add_rotor_inertia=add_rotor_inertia,
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
            log_check_point_callback=optim._extract_and_log_optimization_result,
        )

        bspline_traj_attributes = optim._extract_bspline_trajectory_attributes(x_val)
        return bspline_traj_attributes
