import argparse
import logging

from pathlib import Path
from typing import List

import numpy as np

from robot_payload_id.utils import JointData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="The directory containing the raw joint data folders to be concatenated.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The directory to save the concatenated joint data to.",
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

    sample_period = joint_datas[0].sample_times_s[1] - joint_datas[0].sample_times_s[0]
    for jd in joint_datas:
        assert np.allclose(
            jd.sample_times_s[1:] - jd.sample_times_s[:-1], sample_period
        ), "All samples must be equally spaced."

    concatenated_joint_data = JointData(
        joint_positions=np.concatenate(
            [jd.joint_positions for jd in joint_datas], axis=0
        ),
        joint_velocities=np.concatenate(
            [jd.joint_velocities for jd in joint_datas], axis=0
        ),
        joint_accelerations=np.concatenate(
            [jd.joint_accelerations for jd in joint_datas], axis=0
        ),
        joint_torques=np.concatenate([jd.joint_torques for jd in joint_datas], axis=0),
        sample_times_s=None,
    )
    concatenated_joint_data.sample_times_s = np.arange(
        0.0, len(concatenated_joint_data.joint_positions) * sample_period, sample_period
    )

    logging.info(f"Saving concatenated joint data to {output_dir}.")
    concatenated_joint_data.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
