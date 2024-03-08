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
        joint_data_raw = JointData.load_from_disk(dir)
        joint_data_raw.joint_accelerations = (
            np.zeros_like(joint_data_raw.joint_positions) * np.nan
        )
        joint_data = joint_data_raw.remove_duplicate_samples()
        joint_datas.append(joint_data)

    averaged_joint_data = JointData(
        joint_positions=np.mean([jd.joint_positions for jd in joint_datas], axis=0),
        joint_velocities=np.mean([jd.joint_velocities for jd in joint_datas], axis=0),
        joint_accelerations=np.mean(
            [jd.joint_accelerations for jd in joint_datas], axis=0
        ),
        joint_torques=np.mean([jd.joint_torques for jd in joint_datas], axis=0),
        sample_times_s=joint_datas[0].sample_times_s,
    )

    # Compute average joint position variance across data sets
    joint_positions_list = [jd.joint_positions for jd in joint_datas]
    variances = np.var(joint_positions_list, axis=0)
    mean_variance_across_time = np.mean(variances, axis=0)
    logging.info(
        f"Mean position variance across time for each joint: {mean_variance_across_time}"
    )

    # Plot all joint positions and averaged positions on the same plot
    # TODO: Do the same for velocities, accelerations, and torques
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

    logging.info(f"Saving averaged joint data to {output_dir}.")
    averaged_joint_data.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
