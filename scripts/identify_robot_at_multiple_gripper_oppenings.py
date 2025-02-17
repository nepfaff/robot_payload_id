"""Identifies the robot parameters at multiple gripper openings.

The script will identify the robot parameters at multiple gripper openings. It will
load the joint data from the given path, process the joint data, and then identify the
robot parameters.


The expected input directory structure is as follows:

<joint_data_path>/gripper_position_<gripper_position>/
    run_<run_idx>/
        joint_positions.npy
        joint_torques.npy
        sample_times_s.npy

The identified parameters will be saved into
<joint_data_path>/gripper_position_<gripper_position>/identified_robot_params.npy.
"""

import argparse
import logging

from pathlib import Path

import numpy as np

from robot_payload_id.data import (
    compute_base_param_mapping,
    extract_numeric_data_matrix_autodiff,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import solve_inertial_param_sdp
from robot_payload_id.utils import (
    ArmPlantComponents,
    JointData,
    get_plant_joint_params,
    process_joint_data,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_data_points",
        type=int,
        default=10000,
        help="Number of data points to use.",
    )
    parser.add_argument(
        "--joint_data_path",
        type=Path,
        help="See main file docstring.",
    )
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=1e-3,
        help="The regularization weight.",
    )
    parser.add_argument(
        "--pos_order",
        type=int,
        default=10,
        help="The order of the filter for the joint positions. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--pos_cutoff_freq_hz",
        type=float,
        default=60.0,
        help="The cutoff frequency of the filter for the joint positions. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--vel_order",
        type=int,
        default=10,
        help="The order of the filter for the joint velocities. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--vel_cutoff_freq_hz",
        type=float,
        default=5.6,
        help="The cutoff frequency of the filter for the joint velocities. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--acc_order",
        type=int,
        default=10,
        help="The order of the filter for the joint accelerations. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--acc_cutoff_freq_hz",
        type=float,
        default=4.2,
        help="The cutoff frequency of the filter for the joint accelerations. Only used "
        + "if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_order",
        type=int,
        default=10,
        help="The order of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_cutoff_freq_hz",
        type=float,
        default=4.0,
        help="The cutoff frequency of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--only_save_averaged_joint_data",
        action="store_true",
        help="Whether to only save the averaged joint data and not identify the "
        + "robot parameters.",
    )
    parser.add_argument(
        "--time_to_cutoff_at_beginning_s",
        type=float,
        default=0.0,
        help="The time to cutoff at the beginning of the data.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )

    args = parser.parse_args()
    regularization_weight = args.regularization_weight
    pos_filter_order = args.pos_order
    pos_cutoff_freq_hz = args.pos_cutoff_freq_hz
    vel_filter_order = args.vel_order
    vel_cutoff_freq_hz = args.vel_cutoff_freq_hz
    acc_filter_order = args.acc_order
    acc_cutoff_freq_hz = args.acc_cutoff_freq_hz
    torque_filter_order = args.torque_order
    torque_cutoff_freq_hz = args.torque_cutoff_freq_hz
    only_save_averaged_joint_data = args.only_save_averaged_joint_data
    time_to_cutoff_at_beginning_s = args.time_to_cutoff_at_beginning_s

    logging.basicConfig(level=args.log_level)

    # Create arm
    num_joints = 7
    # NOTE: This model must not have a payload attached. Otherwise, the w0 term will be
    # wrong and include the payload inertia.
    model_path = "./models/iiwa.dmd.yaml"
    arm_components = create_arm(
        arm_file_path=model_path, num_joints=num_joints, time_step=0.0
    )
    arm_plant_components = ArmPlantComponents(
        plant=arm_components.plant,
        plant_context=arm_components.plant.CreateDefaultContext(),
    )

    # Compute the base parameter mapping
    num_random_points = 2000
    joint_data_random = JointData(
        joint_positions=np.random.rand(num_random_points, num_joints) - 0.5,
        joint_velocities=np.random.rand(num_random_points, num_joints) - 0.5,
        joint_accelerations=np.random.rand(num_random_points, num_joints) - 0.5,
        joint_torques=np.zeros((num_random_points, num_joints)),
        sample_times_s=np.zeros(num_random_points),
    )
    W_data_random, _, _ = extract_numeric_data_matrix_autodiff(
        plant_components=arm_plant_components,
        joint_data=joint_data_random,
        add_rotor_inertia=False,
        add_reflected_inertia=True,
        add_viscous_friction=True,
        add_dynamic_dry_friction=True,
        payload_only=False,
    )
    base_param_mapping = compute_base_param_mapping(W_data_random)
    logging.info(
        f"{base_param_mapping.shape[1]} out of {base_param_mapping.shape[0]} "
        + "parameters are identifiable."
    )

    subdirs = [d for d in args.joint_data_path.iterdir() if d.is_dir()]
    for subdir in subdirs:
        logging.info(f"Processing {subdir}")

        # Load all runs and average data over time.
        joint_datas = [
            JointData.load_from_disk_allow_missing(run_dir)
            for run_dir in subdir.iterdir()
            if run_dir.is_dir() and run_dir.name.startswith("run_")
        ]
        raw_joint_data = JointData.average_joint_datas(joint_datas)
        raw_joint_data = JointData.cut_off_at_beginning(
            raw_joint_data, time_to_cutoff_at_beginning_s
        )

        # Save the averaged joint data.
        out_path = subdir / "averaged_joint_data"
        logging.info(f"Saving averaged joint data to {out_path}")
        raw_joint_data.save_to_disk(out_path)

        if only_save_averaged_joint_data:
            continue

        # Process joint data
        joint_data = process_joint_data(
            joint_data=raw_joint_data,
            num_endpoints_to_remove=0,
            compute_velocities=True,
            filter_positions=False,
            pos_filter_order=pos_filter_order,
            pos_cutoff_freq_hz=pos_cutoff_freq_hz,
            vel_filter_order=vel_filter_order,
            vel_cutoff_freq_hz=vel_cutoff_freq_hz,
            acc_filter_order=acc_filter_order,
            acc_cutoff_freq_hz=acc_cutoff_freq_hz,
            torque_filter_order=torque_filter_order,
            torque_cutoff_freq_hz=torque_cutoff_freq_hz,
        )

        # Generate data matrix
        (W_data_raw, w0_data, _) = extract_numeric_data_matrix_autodiff(
            plant_components=arm_plant_components,
            joint_data=joint_data,
            add_rotor_inertia=False,
            add_reflected_inertia=True,
            add_viscous_friction=True,
            add_dynamic_dry_friction=True,
            payload_only=False,
        )
        tau_data = joint_data.joint_torques.flatten()
        # Transform from affine `tau = W * params + w0` into linear `(tau - w0) = W * params`
        tau_data -= w0_data

        # Remove structurally unidentifiable columns to prevent SolutionResult.kUnbounded
        W_data = np.empty((W_data_raw.shape[0], base_param_mapping.shape[1]))
        for i in range(args.num_data_points):
            W_data[i * num_joints : (i + 1) * num_joints, :] = (
                W_data_raw[i * num_joints : (i + 1) * num_joints, :]
                @ base_param_mapping
            )

        # Construct initial parameter guess
        params_guess = get_plant_joint_params(
            arm_components.plant,
            arm_components.plant.CreateDefaultContext(),
            add_rotor_inertia=False,
            add_reflected_inertia=True,
            add_viscous_friction=True,
            add_dynamic_dry_friction=True,
            payload_only=False,
        )

        # Solve the SDP
        _, result, variable_names, variable_vec, _ = solve_inertial_param_sdp(
            num_links=num_joints,
            W_data=W_data,
            tau_data=tau_data,
            base_param_mapping=base_param_mapping,
            regularization_weight=regularization_weight,
            params_guess=params_guess,
            use_euclidean_regularization=False,
            identify_rotor_inertia=False,
            identify_reflected_inertia=True,
            identify_viscous_friction=True,
            identify_dynamic_dry_friction=True,
            payload_only=False,
        )
        if result.is_success():
            final_cost = result.get_optimal_cost()
            logging.info(f"SDP cost: {final_cost}")
            var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_vec)))

            out_path = subdir / "identified_robot_params.npy"
            logging.info(f"Saving parameters to {out_path}")
            np.save(out_path, var_sol_dict)
        else:
            logging.warning("Failed to solve inertial parameter SDP!")
            logging.info(f"Solution result:\n{result.get_solution_result()}")
            logging.info(f"Solver details:\n{result.get_solver_details()}")


if __name__ == "__main__":
    main()
