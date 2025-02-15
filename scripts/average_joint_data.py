import argparse
import logging

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from robot_payload_id.utils import JointData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing the raw joint data folders. Each folder in this "
        + "directory should contain joint data.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The directory to save the processed joint data to.",
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

    joint_datas: List[JointData] = []
    for dir in input_dir.iterdir():
        if not dir.is_dir():
            continue
        logging.info(f"Loading joint data from {dir}.")
        joint_data_raw = JointData.load_from_disk_allow_missing(dir)
        joint_data_raw.joint_accelerations = (
            np.zeros_like(joint_data_raw.joint_positions) * np.nan
        )
        joint_data = joint_data_raw.remove_duplicate_samples()
        joint_datas.append(joint_data)

    # Remove all non-common samples
    min_length = min([len(jd.sample_times_s) for jd in joint_datas])
    for jd in joint_datas:
        jd.joint_positions = jd.joint_positions[:min_length]
        if jd.joint_velocities is not None:
            jd.joint_velocities = jd.joint_velocities[:min_length]
        if jd.joint_accelerations is not None:
            jd.joint_accelerations = jd.joint_accelerations[:min_length]
        jd.joint_torques = jd.joint_torques[:min_length]
        jd.sample_times_s = jd.sample_times_s[:min_length]

    # Validate that all samples are equally spaced
    sample_period = joint_datas[0].sample_times_s[1] - joint_datas[0].sample_times_s[0]
    for jd in joint_datas:
        assert np.allclose(
            jd.sample_times_s[1:] - jd.sample_times_s[:-1], sample_period
        ), "Sample times are not equally spaced."
    logging.info(f"Sample period: {sample_period} seconds.")

    averaged_joint_data = JointData.average_joint_datas(joint_datas)

    # Compute average joint position variance across data sets
    joint_positions_list = [jd.joint_positions for jd in joint_datas]
    variances = np.var(joint_positions_list, axis=0)
    mean_variance_across_time = np.mean(variances, axis=0)
    logging.info(
        f"Mean position variance across time for each joint: {mean_variance_across_time}"
    )

    if joint_datas[0].joint_velocities is not None:
        # Compute average joint velocity variance across data sets
        joint_velocities_list = [jd.joint_velocities for jd in joint_datas]
        variances = np.var(joint_velocities_list, axis=0)
        mean_variance_across_time = np.mean(variances, axis=0)
        logging.info(
            f"Mean velocity variance across time for each joint: {mean_variance_across_time}"
        )

    # Compute average joint torque variance across data sets
    joint_torques_list = [jd.joint_torques for jd in joint_datas]
    variances = np.var(joint_torques_list, axis=0)
    mean_variance_across_time = np.mean(variances, axis=0)
    logging.info(
        f"Mean torque variance across time for each joint: {mean_variance_across_time}"
    )

    # TODO: Refactor the plotting to reduce code duplication
    # Plot all joint positions and averaged positions on the same plot
    num_joints = joint_datas[0].joint_positions.shape[1]
    fig, axes = plt.subplots(
        nrows=num_joints, ncols=1, figsize=(num_joints * 7, 15), sharex=True
    )
    for i in range(num_joints):
        ax = axes[i]
        for joint_data_raw in joint_datas:
            ax.plot(joint_data_raw.sample_times_s, joint_data_raw.joint_positions[:, i])
        ax.plot(
            averaged_joint_data.sample_times_s,
            averaged_joint_data.joint_positions[:, i],
            color="red",
            label="Averaged",
        )
        ax.set_title(f"Joint {i+1}")
        if i == 0:  # Add a legend to the first subplot
            ax.legend()
    # Set a common x-label
    fig.text(
        0.5,
        0.04,
        "Sample Times (s)",
        ha="center",
        fontsize=12,
    )
    # Set a common y-label
    fig.text(
        0.04,
        0.5,
        "Joint positions (rad)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    plt.show()

    if joint_datas[0].joint_velocities is not None:
        # Plot all joint velocities and averaged velocities on the same plot
        fig, axes = plt.subplots(
            nrows=num_joints, ncols=1, figsize=(num_joints * 7, 15), sharex=True
        )
        for i in range(num_joints):
            ax = axes[i]
            for joint_data_raw in joint_datas:
                ax.plot(
                    joint_data_raw.sample_times_s, joint_data_raw.joint_velocities[:, i]
                )
            ax.plot(
                averaged_joint_data.sample_times_s,
                averaged_joint_data.joint_velocities[:, i],
                color="red",
                label="Averaged",
            )
            ax.set_title(f"Joint {i+1}")
            if i == 0:  # Add a legend to the first subplot
                ax.legend()
        # Set a common x-label
        fig.text(
            0.5,
            0.04,
            "Sample Times (s)",
            ha="center",
            fontsize=12,
        )
        # Set a common y-label
        fig.text(
            0.04,
            0.5,
            "Joint velocities (rad/s)",
            va="center",
            rotation="vertical",
            fontsize=12,
        )
        plt.show()

    # Plot all joint torques and averaged torques on the same plot
    fig, axes = plt.subplots(
        nrows=num_joints, ncols=1, figsize=(num_joints * 7, 15), sharex=True
    )
    for i in range(num_joints):
        ax = axes[i]
        for joint_data_raw in joint_datas:
            ax.plot(joint_data_raw.sample_times_s, joint_data_raw.joint_torques[:, i])
        ax.plot(
            averaged_joint_data.sample_times_s,
            averaged_joint_data.joint_torques[:, i],
            color="red",
            label="Averaged",
        )
        ax.set_title(f"Joint {i+1}")
        if i == 0:  # Add a legend to the first subplot
            ax.legend()
    # Set a common x-label
    fig.text(
        0.5,
        0.04,
        "Sample Times (s)",
        ha="center",
        fontsize=12,
    )
    # Set a common y-label
    fig.text(
        0.04,
        0.5,
        "Torque (Nm)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    plt.show()

    logging.info(f"Saving averaged joint data to {output_dir}.")
    averaged_joint_data.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
