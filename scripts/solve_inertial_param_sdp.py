import argparse
import logging

import numpy as np

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_simple_sinusoidal_traj_params,
    extract_data_matrix_autodiff,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import solve_inertial_param_sdp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help="Use one link arm instead of 7-DOF arm.",
    )
    parser.add_argument(
        "--remove_unidentifiable_params",
        action="store_true",
        help="Remove unidentifiable parameters instead of identifying all parameters.",
    )
    parser.add_argument(
        "--num_data_points",
        type=int,
        default=50000,
        help="Number of data points to use.",
    )
    parser.add_argument(
        "--qr_tolerance",
        type=float,
        default=1e-12,
        help="Tolerance for QR decomposition.",
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

    # Generate data matrix
    # TODO: Make configurable (parameterization + parameters)
    joint_data = compute_autodiff_joint_data_from_simple_sinusoidal_traj_params(
        plant=arm_components.plant,
        num_timesteps=args.num_data_points,
        timestep=1.0,
        a=-40.2049 * np.ones(num_joints),
        b=20.8 * np.zeros(num_joints),
        omega=0.5,
    )
    W_data, tau_data = extract_data_matrix_autodiff(
        arm_components=arm_components, joint_data=joint_data
    )

    identifiable = None
    if args.remove_unidentifiable_params:
        # Identify the identifiable parameters using the QR decomposition
        _, R = np.linalg.qr(W_data)
        identifiable = np.abs(np.diag(R)) > args.qr_tolerance

    prog, result, variable_names, variable_list = solve_inertial_param_sdp(
        num_links=num_joints,
        W_data=W_data,
        tau_data=tau_data,
        identifiable=identifiable,
    )
    if result.is_success():
        var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_list)))
        logging.info(f"SDP result:\n{var_sol_dict}")
    else:
        logging.warn("Failed to solve inertial parameter SDP!")
        logging.info(f"MathematicalProgram:\n{prog}")
        logging.info(f"Solution result:\n{result.get_solution_result()}")
        logging.info(f"Solver details:\n{result.get_solver_details()}")


if __name__ == "__main__":
    main()
