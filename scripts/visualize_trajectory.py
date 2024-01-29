import argparse

from pathlib import Path

import numpy as np

from pydrake.all import Simulator

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.environment import create_arm


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
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy'. If "
        + "--remove_unidentifiable_params is set, the folder must also contain "
        + "'base_param_mapping.npy'.",
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

    # Create arm
    num_joints = 1 if args.use_one_link_arm else 7
    urdf_path = (
        "./models/one_link_arm.urdf"
        if args.use_one_link_arm
        else "./models/iiwa.dmd.yaml"
    )
    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=0.0
    )

    # Load trajectory parameters
    traj_parameter_path = args.traj_parameter_path
    a_data = np.load(traj_parameter_path / "a_value.npy").reshape((num_joints, -1))
    b_data = np.load(traj_parameter_path / "b_value.npy").reshape((num_joints, -1))
    q0_data = np.load(traj_parameter_path / "q0_value.npy")

    joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
        plant=arm_components.plant,
        num_timesteps=args.num_timesteps,
        time_horizon=args.time_horizon,
        a=a_data,
        b=b_data,
        q0=q0_data,
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
