import argparse

from pathlib import Path

import numpy as np

from pydrake.all import (
    BsplineBasis_,
    BsplineTrajectory_,
    Expression,
    KinematicTrajectoryOptimization,
    Solve,
)

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import BsplineTrajectoryAttributes
from robot_payload_id.utils import JointData


def extract_bspline_trajectory_attributes(
    var_values: np.ndarray,
    num_control_points: int,
    num_joints: int,
    trajopt: KinematicTrajectoryOptimization,
) -> BsplineTrajectoryAttributes:
    """Extracts the B-spline trajectory attributes from the decision variable values."""
    control_points = (
        np.array(var_values[:-1]).reshape((num_control_points, num_joints)).T
    )
    traj_duration = np.abs(var_values[-1])
    scaled_knots = np.array(trajopt.basis().knots()) * traj_duration
    return BsplineTrajectoryAttributes(
        spline_order=trajopt.basis().order(),
        control_points=control_points,
        knots=scaled_knots,
    )


def construct_and_sample_traj_sym(
    var_values: np.ndarray,
    num_timesteps: int,
    num_control_points: int,
    num_joints: int,
    trajopt: KinematicTrajectoryOptimization,
) -> JointData:
    # Construct the trajectory
    bspline_traj_attributes = extract_bspline_trajectory_attributes(
        var_values=var_values,
        num_control_points=num_control_points,
        num_joints=num_joints,
        trajopt=trajopt,
    )
    traj = BsplineTrajectory_[Expression](
        basis=BsplineBasis_[Expression](
            bspline_traj_attributes.spline_order, trajopt.basis().knots()
        ),
        control_points=bspline_traj_attributes.control_points,
    )

    # Sample the trajectory
    q_numeric = np.empty((num_timesteps, num_joints), dtype=Expression)
    q_dot_numeric = np.empty((num_timesteps, num_joints), dtype=Expression)
    q_ddot_numeric = np.empty((num_timesteps, num_joints), dtype=Expression)
    sample_times_s = np.linspace(0.0, 1.0, num=num_timesteps)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help="Use one link arm instead of 7-DOF arm.",
    )
    parser.add_argument(
        "--traj_parameter_path",
        type=Path,
        required=True,
        help="Path to the trajectory parameter folder. The folder must contain "
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy'",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="The directory to save the B-spline trajectory to.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=100,
        help="The number of timesteps to use. This should correspond to the optimal "
        + "experiment design parameter.",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=10.0,
        help="The time horizon/ duration of the trajectory. This should correspond to "
        + "the optimal experiment design parameter.",
    )
    parser.add_argument(
        "--num_control_points_initial",
        type=int,
        default=100,
        help="The initial number of control points to use for the B-spline trajectory.",
    )
    parser.add_argument(
        "--num_control_points_step",
        type=int,
        default=5,
        help="The step size for the number of control points to use for the B-spline "
        + "trajectory.",
    )
    parser.add_argument(
        "--spline_order",
        type=int,
        default=4,
        help="The order of the B-spline basis functions.",
    )
    parser.add_argument(
        "--match_velocity",
        action="store_true",
        help="Match the velocity of the Fourier series trajectory.",
    )
    parser.add_argument(
        "--match_acceleration",
        action="store_true",
        help="Match the acceleration of the Fourier series trajectory.",
    )
    parser.add_argument(
        "--use_constraints",
        action="store_true",
        help="Whether to solve the problem using constraints rather than costs. Note "
        + "that this requires more control points and is thus not recommended.",
    )

    args = parser.parse_args()
    num_timesteps = args.num_timesteps
    traj_parameter_path = args.traj_parameter_path
    num_control_points_initial = args.num_control_points_initial
    num_control_points_step = args.num_control_points_step
    spline_order = args.spline_order
    trajectory_duration = args.time_horizon
    match_velocity = args.match_velocity
    match_acceleration = args.match_acceleration
    save_dir: Path = args.save_dir
    use_constraints = args.use_constraints

    min_num_control_points = (
        (
            num_timesteps
            + match_velocity * num_timesteps
            + match_acceleration * num_timesteps
        )
        if use_constraints
        else 2
    )
    assert num_control_points_initial >= min_num_control_points

    # Create arm
    num_joints = 1 if args.use_one_link_arm else 7
    urdf_path = (
        "./models/one_link_arm_with_obstacle.dmd.yaml"
        if args.use_one_link_arm
        else "./models/iiwa_with_obstacles.dmd.yaml"
    )
    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=0.0
    )

    # Load Fourier series parameters and compute joint data
    a_data = np.load(traj_parameter_path / "a_value.npy").reshape(
        (num_joints, -1), order="F"
    )
    b_data = np.load(traj_parameter_path / "b_value.npy").reshape(
        (num_joints, -1), order="F"
    )
    q0_data = np.load(traj_parameter_path / "q0_value.npy")
    joint_data_gt = compute_autodiff_joint_data_from_fourier_series_traj_params1(
        plant=arm_components.plant,
        num_timesteps=num_timesteps,
        time_horizon=trajectory_duration,
        a=a_data,
        b=b_data,
        q0=q0_data,
    )

    num_control_points = num_control_points_initial
    is_success = False
    iters = 0
    while not is_success:
        if use_constraints:
            trajopt = KinematicTrajectoryOptimization(
                num_positions=num_joints,
                num_control_points=num_control_points,
                spline_order=spline_order,
                duration=trajectory_duration,
            )
            prog = trajopt.prog()
            s_samples = joint_data_gt.sample_times_s / joint_data_gt.sample_times_s[-1]
            for pos, vel, acc, s_sample in zip(
                joint_data_gt.joint_positions,
                joint_data_gt.joint_velocities,
                joint_data_gt.joint_accelerations,
                s_samples,
            ):
                trajopt.AddPathPositionConstraint(lb=pos, ub=pos, s=s_sample)
                # NOTE: Velocity and acceleration constraints don't directly
                # correspond to actual velocities/ accelerations due to time scaling.
                if match_velocity:
                    trajopt.AddPathVelocityConstraint(lb=vel, ub=vel, s=s_sample)
                if match_acceleration:
                    trajopt.AddPathAccelerationConstraint(lb=acc, ub=acc, s=s_sample)
        else:
            trajopt = KinematicTrajectoryOptimization(
                num_positions=num_joints,
                num_control_points=num_control_points,
                spline_order=spline_order,
                duration=trajectory_duration,
            )
            prog = trajopt.get_mutable_prog()
            joint_data_sym = construct_and_sample_traj_sym(
                var_values=prog.decision_variables(),
                num_timesteps=num_timesteps,
                num_control_points=num_control_points,
                num_joints=num_joints,
                trajopt=KinematicTrajectoryOptimization(
                    num_positions=num_joints,
                    num_control_points=num_control_points,
                    spline_order=spline_order,
                    duration=trajectory_duration,
                ),
            )
            for pos, pos_gt, vel, vel_gt, acc, acc_gt in zip(
                joint_data_sym.joint_positions,
                joint_data_gt.joint_positions,
                joint_data_sym.joint_velocities,
                joint_data_gt.joint_velocities,
                joint_data_sym.joint_accelerations,
                joint_data_gt.joint_accelerations,
            ):
                prog.AddQuadraticCost((pos - pos_gt).T @ (pos - pos_gt))
                # NOTE: Velocity and acceleration costs don't directly
                # correspond to actual velocities/ accelerations due to time scaling
                if match_velocity:
                    prog.AddQuadraticCost((vel - vel_gt).T @ (vel - vel_gt))
                if match_acceleration:
                    prog.AddQuadraticCost((acc - acc_gt).T @ (acc - acc_gt))

        result = Solve(prog)
        is_success = result.is_success()

        num_control_points += num_control_points_step
        iters += 1

    final_num_control_points = num_control_points - num_control_points_step
    print(f"Final number of control points: {final_num_control_points}")
    print(f"Number of iterations: {iters}")
    print(
        "Optimal cost (increase number of control points to decrease): "
        + f"{result.get_optimal_cost()}"
    )

    # Save the trajectory
    save_dir.mkdir(parents=True, exist_ok=True)
    if use_constraints:
        extract_bspline_trajectory_attributes(
            var_values=result.GetSolution(),
            num_control_points=final_num_control_points,
            num_joints=num_joints,
            trajopt=trajopt,
        ).log(save_dir)
    else:
        scaled_knots = np.array(trajopt.basis().knots()) * trajectory_duration
        control_points = result.GetSolution(trajopt.control_points())
        BsplineTrajectoryAttributes(
            spline_order=trajopt.basis().order(),
            control_points=control_points,
            knots=scaled_knots,
        ).log(save_dir)


if __name__ == "__main__":
    main()
