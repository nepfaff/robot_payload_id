import argparse
import logging

from pathlib import Path

from robot_payload_id.data import (
    extract_symbolic_data_matrix,
    pickle_symbolic_data_matrix,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_symbolic_plant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help="Use one link arm instead of 7-DOF arm.",
    )
    parser.add_argument(
        "--simplify_before_differentiation",
        action="store_true",
        help="Simplify symbolic expression with sumpy before differentiation.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    use_one_link_arm = args.use_one_link_arm
    urdf_path = (
        "./models/one_link_arm.urdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )
    num_joints = 1 if use_one_link_arm else 7
    save_dir = Path(
        "data/symbolic_data_matrix_one_link_arm"
        if use_one_link_arm
        else "data/symbolic_data_matrix_iiwa"
    )

    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=0.0
    )
    sym_plant_components = create_symbolic_plant(
        arm_components=arm_components, use_lumped_parameters=True
    )

    W_sym = extract_symbolic_data_matrix(
        symbolic_plant_components=sym_plant_components,
        simplify=args.simplify_before_differentiation,
    )
    pickle_symbolic_data_matrix(
        W_sym=W_sym,
        state_variables=sym_plant_components.state_variables,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
