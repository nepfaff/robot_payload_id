import argparse

from pathlib import Path

import numpy as np

from pydrake.all import KinematicTrajectoryOptimization, Solve

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import BsplineTrajectoryAttributes


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

    args = parser.parse_args()
    num_timesteps = args.num_timesteps
    traj_parameter_path = args.traj_parameter_path
    num_control_points_initial = args.num_control_points_initial
    num_control_points_step = args.num_control_points_step
    spline_order = args.spline_order
    trajectory_duration = args.time_horizon
    match_velocity = args.match_velocity
    match_acceleration = args.match_acceleration
    save_dir = args.save_dir

    # Mosek fails with assertion error if num_control_points < min_num_control_points
    min_num_control_points = (
        num_timesteps
        + match_velocity * num_timesteps
        + match_acceleration * num_timesteps
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
        arm_file_path=urdf_path, num_joints=num_joints, time_step=0.0, use_meshcat=True
    )

    # Load Fourier series parameters and compute joint data
    a_data = np.load(traj_parameter_path / "a_value.npy").reshape(
        (num_joints, -1), order="F"
    )
    b_data = np.load(traj_parameter_path / "b_value.npy").reshape(
        (num_joints, -1), order="F"
    )
    q0_data = np.load(traj_parameter_path / "q0_value.npy")
    joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
        plant=arm_components.plant,
        num_timesteps=num_timesteps,
        time_horizon=args.time_horizon,
        a=a_data,
        b=b_data,
        q0=q0_data,
    )

    num_control_points = num_control_points_initial
    is_success = False
    iters = 0
    while not is_success:
        trajopt = KinematicTrajectoryOptimization(
            num_positions=num_joints,
            num_control_points=num_control_points,
            spline_order=spline_order,
            duration=trajectory_duration,
        )
        s_samples = joint_data.sample_times_s / joint_data.sample_times_s[-1]
        for pos, vel, acc, s_sample in zip(
            joint_data.joint_positions,
            joint_data.joint_velocities,
            joint_data.joint_accelerations,
            s_samples,
        ):
            trajopt.AddPathPositionConstraint(lb=pos, ub=pos, s=s_sample)
            if match_velocity:
                trajopt.AddPathVelocityConstraint(lb=vel, ub=vel, s=s_sample)
            if match_acceleration:
                trajopt.AddPathAccelerationConstraint(lb=acc, ub=acc, s=s_sample)

        result = Solve(prog=trajopt.prog())
        is_success = result.is_success()

        num_control_points += num_control_points_step
        iters += 1

    print(
        f"Final number of control points: {num_control_points-num_control_points_step}"
    )
    print(f"Number of iterations: {iters}")

    # Save the trajectory
    save_dir.mkdir(parents=True, exist_ok=True)
    scaled_knots = np.array(trajopt.basis().knots()) * trajectory_duration
    control_points = result.GetSolution(trajopt.control_points())
    BsplineTrajectoryAttributes(
        spline_order=trajopt.basis().order(),
        control_points=control_points,
        knots=scaled_knots,
    ).log(save_dir)


if __name__ == "__main__":
    main()
