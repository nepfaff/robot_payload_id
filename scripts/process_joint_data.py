import argparse
import logging

from pathlib import Path

from robot_payload_id.utils import JointData, process_joint_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing the raw joint data.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The directory to save the processed joint data to.",
    )
    parser.add_argument(
        "--compute_velocities",
        action="store_true",
        help="Whether to compute velocities from the positions rather than taking the "
        + "ones in `joint_data`.",
    )
    parser.add_argument(
        "--filter_positions",
        action="store_true",
        help="Whether to filter the joint positions.",
    )
    parser.add_argument(
        "--pos_order",
        type=int,
        default=10,
        help="The order of the filter for the joint positions.",
    )
    parser.add_argument(
        "--pos_cutoff_freq_hz",
        type=float,
        default=60.0,
        help="The cutoff frequency of the filter for the joint positions.",
    )
    parser.add_argument(
        "--vel_order",
        type=int,
        default=10,
        help="The order of the filter for the joint velocities.",
    )
    parser.add_argument(
        "--vel_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the joint velocities.",
    )
    parser.add_argument(
        "--acc_order",
        type=int,
        default=10,
        help="The order of the filter for the joint accelerations.",
    )
    parser.add_argument(
        "--acc_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the joint accelerations.",
    )
    parser.add_argument(
        "--torque_order",
        type=int,
        default=10,
        help="The order of the filter for the joint torques.",
    )
    parser.add_argument(
        "--torque_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the joint torques.",
    )
    parser.add_argument(
        "--ft_sensor_force_order",
        type=int,
        default=10,
        help="The order of the filter for the force from the FT sensor.",
    )
    parser.add_argument(
        "--ft_sensor_force_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the force from the FT sensor.",
    )
    parser.add_argument(
        "--ft_sensor_torque_order",
        type=int,
        default=10,
        help="The order of the filter for the torque from the FT sensor.",
    )
    parser.add_argument(
        "--ft_sensor_torque_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the torque from the FT sensor.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    logging.basicConfig(level=args.log_level)

    logging.info(f"Loading joint data from {input_dir}.")
    raw_joint_data = JointData.load_from_disk(input_dir)

    processed_joint_data = process_joint_data(
        joint_data=raw_joint_data,
        compute_velocities=args.compute_velocities,
        filter_positions=args.filter_positions,
        pos_filter_order=args.pos_order,
        pos_cutoff_freq_hz=args.pos_cutoff_freq_hz,
        vel_filter_order=args.vel_order,
        vel_cutoff_freq_hz=args.vel_cutoff_freq_hz,
        acc_filter_order=args.acc_order,
        acc_cutoff_freq_hz=args.acc_cutoff_freq_hz,
        torque_filter_order=args.torque_order,
        torque_cutoff_freq_hz=args.torque_cutoff_freq_hz,
        ft_sensor_force_order=args.ft_sensor_force_order,
        ft_sensor_force_cutoff_freq_hz=args.ft_sensor_force_cutoff_freq_hz,
        ft_sensor_torque_order=args.ft_sensor_torque_order,
        ft_sensor_torque_cutoff_freq_hz=args.ft_sensor_torque_cutoff_freq_hz,
    )

    logging.info(f"Saving processed joint data to {output_dir}.")
    processed_joint_data.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
