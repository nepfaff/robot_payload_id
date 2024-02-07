import argparse
import os

from pathlib import Path

import numpy as np

from pydrake.all import BsplineBasis, BsplineTrajectory, Simulator

import wandb

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import BsplineTrajectoryAttributes
from robot_payload_id.utils import JointData


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
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
        + "'knots.npy', and 'spline_order.npy'.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
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
        "--use_wandb",
        action="store_true",
        help="Use Weights and Biases for logging.",
    )
    args = parser.parse_args()
    num_timesteps = args.num_timesteps
    traj_parameter_path = args.traj_parameter_path

    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(
            project="robot_payload_id_visualized_trajectories",
            name=f"visualized_trajectory: {str(traj_parameter_path)}",
            config=vars(args),
        )

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

    # Load trajectory parameters
    is_fourier_series = os.path.exists(traj_parameter_path / "a_value.npy")
    if is_fourier_series:
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
    else:
        traj_attrs = BsplineTrajectoryAttributes.load(traj_parameter_path)
        traj = BsplineTrajectory(
            basis=BsplineBasis(order=traj_attrs.spline_order, knots=traj_attrs.knots),
            control_points=traj_attrs.control_points,
        )

        # Sample the trajectory
        q_numeric = np.empty((num_timesteps, num_joints))
        q_dot_numeric = np.empty((num_timesteps, num_joints))
        q_ddot_numeric = np.empty((num_timesteps, num_joints))
        sample_times_s = np.linspace(
            traj.start_time(), traj.end_time(), num=num_timesteps
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

    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)

    arm_components.meshcat_visualizer.StartRecording()
    for q, q_dot, t in zip(
        joint_data.joint_positions,
        joint_data.joint_velocities,
        joint_data.sample_times_s,
    ):
        arm_components.plant.SetPositions(plant_context, q)
        arm_components.plant.SetVelocities(plant_context, q_dot)
        simulator.AdvanceTo(t)

    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()

    if use_wandb:
        html = arm_components.meshcat.StaticHtml()
        wandb.log({"trajectory": wandb.Html(html)})


if __name__ == "__main__":
    main()
