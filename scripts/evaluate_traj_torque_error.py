import argparse
import logging

from pathlib import Path

import numpy as np

from robot_payload_id.data import extract_numeric_data_matrix_autodiff
from robot_payload_id.environment import create_arm
from robot_payload_id.utils import (
    ArmPlantComponents,
    JointData,
    process_joint_data,
    write_parameters_to_plant,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_param_path",
        type=Path,
        help="Path to the initial parameter `.npy` file. If not provided, the initial "
        + "parameters are set to the model's parameters.",
    )
    parser.add_argument(
        "--joint_data_path",
        type=Path,
        help="Path to the joint data folder. The folder must contain "
        + "`joint_positions.npy`, `joint_velocities.npy`, `joint_accelerations.npy`, "
        + "`joint_torques.npy`, and `sample_times_s.npy`. NOTE: Only one of "
        + "`--traj_parameter_path` or `--joint_data_path` should be set.",
    )
    parser.add_argument(
        "--process_joint_data",
        action="store_true",
        help="Whether to process the joint data before using it.",
    )
    parser.add_argument(
        "--num_endpoints_to_remove",
        type=int,
        default=5,
        help="The number of endpoints to remove from the beginning and end of the "
        + "trajectory. This is useful as the sample times are not always increasing "
        + "with the same period at the beginning and end of the trajectory. Only used "
        + "if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--vel_order",
        type=int,
        default=20,
        help="The order of the filter for the joint velocities. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--vel_cutoff_freq_hz",
        type=float,
        default=2.0,
        help="The cutoff frequency of the filter for the joint velocities. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--acc_order",
        type=int,
        default=20,
        help="The order of the filter for the joint accelerations. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--acc_cutoff_freq_hz",
        type=float,
        default=2.0,
        help="The cutoff frequency of the filter for the joint accelerations. Only used "
        + "if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_order",
        type=int,
        default=12,
        help="The order of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_cutoff_freq_hz",
        type=float,
        default=1.6,
        help="The cutoff frequency of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()
    initial_param_path = args.initial_param_path
    joint_data_path = args.joint_data_path
    do_process_joint_data = args.process_joint_data
    num_endpoints_to_remove = args.num_endpoints_to_remove
    vel_filter_order = args.vel_order
    vel_cutoff_freq_hz = args.vel_cutoff_freq_hz
    acc_filter_order = args.acc_order
    acc_cutoff_freq_hz = args.acc_cutoff_freq_hz
    torque_filter_order = args.torque_order
    torque_cutoff_freq_hz = args.torque_cutoff_freq_hz

    logging.basicConfig(level=args.log_level)

    # Create the arm
    num_joints = 7
    model_path = "./models/iiwa.dmd.yaml"
    arm_components = create_arm(
        arm_file_path=model_path, num_joints=num_joints, time_step=0.0
    )

    # Load parameters
    if initial_param_path is not None:
        logging.info(f"Loading initial parameters from {initial_param_path}")
        var_sol_dict = np.load(initial_param_path, allow_pickle=True).item()
        arm_plant_components = write_parameters_to_plant(arm_components, var_sol_dict)
        add_rotor_inertia = "rotor_inertia0(0)" in var_sol_dict
        add_reflected_inertia = "reflected_inertia0(0)" in var_sol_dict
        add_viscous_friction = "viscous_friction0(0)" in var_sol_dict
        add_dynamic_dry_friction = "dynamic_dry_friction0(0)" in var_sol_dict
    else:
        arm_plant_components = ArmPlantComponents(
            plant=arm_components.plant,
            plant_context=arm_components.plant.CreateDefaultContext(),
        )
        add_rotor_inertia = False
        add_reflected_inertia = True
        add_viscous_friction = True
        add_dynamic_dry_friction = True

    # Load joint data
    joint_data = JointData.load_from_disk(joint_data_path)

    # Process joint data
    if do_process_joint_data:
        joint_data = process_joint_data(
            joint_data=joint_data,
            num_endpoints_to_remove=num_endpoints_to_remove,
            vel_filter_order=vel_filter_order,
            vel_cutoff_freq_hz=vel_cutoff_freq_hz,
            acc_filter_order=acc_filter_order,
            acc_cutoff_freq_hz=acc_cutoff_freq_hz,
            torque_filter_order=torque_filter_order,
            torque_cutoff_freq_hz=torque_cutoff_freq_hz,
        )

    # Compute model-predicted torques
    (
        _,
        _,
        predicted_torques,
    ) = extract_numeric_data_matrix_autodiff(
        plant_components=arm_plant_components,
        joint_data=joint_data,
        add_rotor_inertia=add_rotor_inertia,
        add_reflected_inertia=add_reflected_inertia,
        add_viscous_friction=add_viscous_friction,
        add_dynamic_dry_friction=add_dynamic_dry_friction,
    )

    # Compute torque error
    num_datapoints = len(joint_data.sample_times_s)
    torque_error = predicted_torques - joint_data.joint_torques.flatten()
    torque_error_sum_normalized = torque_error.T @ torque_error / num_datapoints

    # Print results
    print("Torque error sum normalized: ", torque_error_sum_normalized)


if __name__ == "__main__":
    main()
