import argparse
import os

from pathlib import Path

import numpy as np

from pydrake.all import BsplineBasis, BsplineTrajectory, Simulator

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.environment import create_arm
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
    args = parser.parse_args()
    num_timesteps = args.num_timesteps
    traj_parameter_path = args.traj_parameter_path

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
        a_data = np.load(traj_parameter_path / "a_value.npy").reshape((num_joints, -1))
        b_data = np.load(traj_parameter_path / "b_value.npy").reshape((num_joints, -1))
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
        control_points = np.load(traj_parameter_path / "control_points.npy")
        knots = np.load(traj_parameter_path / "knots.npy")
        spline_order = int(np.load(traj_parameter_path / "spline_order.npy")[0])

        traj = BsplineTrajectory(
            basis=BsplineBasis(order=spline_order, knots=knots),
            control_points=control_points,
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


if __name__ == "__main__":
    main()
