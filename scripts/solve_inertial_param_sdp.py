import argparse
import logging

from pathlib import Path

import numpy as np

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
    extract_numeric_data_matrix_autodiff,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.eric_id.drake_torch_dynamics import (
    calc_inertia_entropic_divergence,
    get_candidate_sys_id_bodies,
    get_plant_inertial_params,
)
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
        "--time_horizon",
        type=float,
        default=10.0,
        required=False,
        help="The time horizon/ duration of the trajectory. The sampling time step is "
        + "computed as time_horizon / num_timesteps.",
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
        "--kPrintToConsole",
        action="store_true",
        help="Whether to print solver output.",
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

    # Load trajectory parameters
    traj_parameter_path = args.traj_parameter_path
    a_data = np.load(traj_parameter_path / "a_value.npy").reshape((num_joints, -1))
    b_data = np.load(traj_parameter_path / "b_value.npy").reshape((num_joints, -1))
    q0_data = np.load(traj_parameter_path / "q0_value.npy")
    base_param_mapping = (
        np.load(traj_parameter_path / "base_param_mapping.npy")
        if args.remove_unidentifiable_params
        else None
    )

    # Generate data matrix
    joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
        plant=arm_components.plant,
        num_timesteps=args.num_data_points,
        time_horizon=args.time_horizon,
        a=a_data,
        b=b_data,
        q0=q0_data,
    )
    W_data_raw, tau_data = extract_numeric_data_matrix_autodiff(
        arm_components=arm_components, joint_data=joint_data
    )

    if base_param_mapping is None:
        W_data = W_data_raw
    else:
        # Remove structurally unidentifiable columns to prevent
        # SolutionResult.kUnbounded
        W_data = np.empty((W_data_raw.shape[0], base_param_mapping.shape[1]))
        for i in range(args.num_data_points):
            W_data[i * num_joints : (i + 1) * num_joints, :] = (
                W_data_raw[i * num_joints : (i + 1) * num_joints, :]
                @ base_param_mapping
            )

    _, result, variable_names, variable_list = solve_inertial_param_sdp(
        num_links=num_joints,
        W_data=W_data,
        tau_data=tau_data,
        base_param_mapping=base_param_mapping,
        solver_kPrintToConsole=args.kPrintToConsole,
    )
    if result.is_success():
        var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_list)))
        logging.info(f"SDP result:\n{var_sol_dict}")

        # Compute entropic divergence to GT parameters
        masses_estiamted = np.array(
            [var_sol_dict[f"m{i}(0)"] for i in range(num_joints)]
        )
        coms_estimated = np.array(
            [
                [
                    var_sol_dict[f"hx{i}(0)"],
                    var_sol_dict[f"hy{i}(0)"],
                    var_sol_dict[f"hz{i}(0)"],
                ]
                / var_sol_dict[f"m{i}(0)"]
                for i in range(num_joints)
            ]
        )
        rot_inertias_estimated = np.array(
            [
                [
                    [
                        var_sol_dict[f"Ixx{i}(0)"],
                        var_sol_dict[f"Ixy{i}(0)"],
                        var_sol_dict[f"Ixz{i}(0)"],
                    ],
                    [
                        var_sol_dict[f"Ixy{i}(0)"],
                        var_sol_dict[f"Iyy{i}(0)"],
                        var_sol_dict[f"Iyz{i}(0)"],
                    ],
                    [
                        var_sol_dict[f"Ixz{i}(0)"],
                        var_sol_dict[f"Iyz{i}(0)"],
                        var_sol_dict[f"Izz{i}(0)"],
                    ],
                ]
                for i in range(num_joints)
            ]
        )
        bodies = get_candidate_sys_id_bodies(arm_components.plant)
        masses_gt, coms_gt, rot_inertias_gt = get_plant_inertial_params(
            arm_components.plant, arm_components.plant.CreateDefaultContext(), bodies
        )
        inertia_entropic_divergence = calc_inertia_entropic_divergence(
            masses_estiamted,
            coms_estimated,
            rot_inertias_estimated,
            masses_gt,
            coms_gt,
            rot_inertias_gt,
        )
        # Zero entropic divergence means the estimated parameters are the same as the
        # ground truth parameters. This is not possible as not all parameters are
        # identifiable.
        logging.info(
            "Inertia entropic divergence from ground truth: "
            + f"{inertia_entropic_divergence}"
        )
        last_link_inertia_entropic_divergence = calc_inertia_entropic_divergence(
            masses_estiamted[-1:],
            coms_estimated[-1:],
            rot_inertias_estimated[-1:],
            masses_gt[-1:],
            coms_gt[-1:],
            rot_inertias_gt[-1:],
        )
        logging.info(
            "Inertia entropic divergence from ground truth for last link: "
            + f"{last_link_inertia_entropic_divergence}"
        )
    else:
        logging.warning("Failed to solve inertial parameter SDP!")
        logging.info(f"Solution result:\n{result.get_solution_result()}")
        logging.info(f"Solver details:\n{result.get_solver_details()}")


if __name__ == "__main__":
    main()
