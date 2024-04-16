import argparse
import copy
import logging
import os

from datetime import datetime
from pathlib import Path

import numpy as np

from pydrake.all import SpatialInertia, UnitInertia

import wandb

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
    compute_ft_sensor_measurements,
    construct_ft_data_matrix,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import solve_ft_payload_sdp
from robot_payload_id.utils import (
    ArmPlantComponents,
    BsplineTrajectoryAttributes,
    FourierSeriesTrajectoryAttributes,
    JointData,
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
        help="Path to the trajectory parameter folder. The folder must contain "
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
        + "'knots.npy', and 'spline_order.npy'. If --remove_unidentifiable_params is "
        + "set, the folder must also contain 'base_param_mapping.npy'. NOTE: Only one "
        + "of `--traj_parameter_path` or `--joint_data_path` should be set.",
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
        "--kPrintToConsole",
        action="store_true",
        help="Whether to print solver output.",
    )
    parser.add_argument(
        "--sensor_body_name",
        type=str,
        default="ft_sensor_link",
        help="The F/T sensor body name. This could be any body with a weld relation "
        + "to the F/T sensor.",
    )
    parser.add_argument(
        "--sensor_frame_name",
        type=str,
        default="ft_sensor_link",
        help="The F/T sensor frame name.",
    )
    parser.add_argument(
        "--payload_frame_name",
        type=str,
        help="The frame to express the identified payload parameters in.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("./models/iiwa7_with_ft_sensor.dmd.yaml"),
        help="Path to the robot model.",
    )
    parser.add_argument(
        "--process_joint_data",
        action="store_true",
        help="Whether to process the joint data before using it.",
    )
    parser.add_argument(
        "--not_compute_velocities",
        action="store_true",
        help="Whether take the velocities in `joint_data` instead of computing them "
        + "from the positions. Only used if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--filter_positions",
        action="store_true",
        help="Whether to filter the joint positions. Only used if `--process_joint_data` "
        + "is set.",
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
        default=10.0,
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
        default=30.0,
        help="The cutoff frequency of the filter for the joint accelerations. Only used "
        + "if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--ft_sensor_force_order",
        type=int,
        default=10,
        help="The order of the filter for the force from the FT sensor. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--ft_sensor_force_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the force from the FT sensor. "
        + "Only used if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--ft_sensor_torque_order",
        type=int,
        default=10,
        help="The order of the filter for the torque from the FT sensor. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--ft_sensor_torque_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the torque from the FT sensor. "
        + "Only used if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["disabled", "online", "offline"],
        default="disabled",
        help="The mode to use for Weights & Biases logging.",
    )
    parser.add_argument(
        "--not_perform_eval",
        action="store_true",
        help="Whether to not perform evaluation after solving the SDP.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )

    args = parser.parse_args()
    traj_parameter_path = args.traj_parameter_path
    joint_data_path = args.joint_data_path
    num_data_points = args.num_data_points
    time_horizon = args.time_horizon
    sensor_body_name = args.sensor_body_name
    sensor_frame_name = args.sensor_frame_name
    payload_frame_name = args.payload_frame_name
    model_path = args.model_path
    do_process_joint_data = args.process_joint_data
    compute_velocities = not args.not_compute_velocities
    filter_positions = args.filter_positions
    pos_filter_order = args.pos_order
    pos_cutoff_freq_hz = args.pos_cutoff_freq_hz
    vel_filter_order = args.vel_order
    vel_cutoff_freq_hz = args.vel_cutoff_freq_hz
    acc_filter_order = args.acc_order
    acc_cutoff_freq_hz = args.acc_cutoff_freq_hz
    ft_sensor_force_order = args.ft_sensor_force_order
    ft_sensor_force_cutoff_freq_hz = args.ft_sensor_force_cutoff_freq_hz
    ft_sensor_torque_order = args.ft_sensor_torque_order
    ft_sensor_torque_cutoff_freq_hz = args.ft_sensor_torque_cutoff_freq_hz
    wandb_mode = args.wandb_mode
    perform_eval = not args.not_perform_eval

    assert (traj_parameter_path is not None) != (joint_data_path is not None), (
        "One but not both of `--traj_parameter_path` and `--joint_data_path` should be "
        + "set."
    )

    logging.basicConfig(level=args.log_level)
    wandb.init(
        project="robot_payload_id",
        name=f"ft_sensor_payload_param_sdp ({datetime.now()})",
        config=vars(args),
        mode=wandb_mode,
    )

    # Create arm
    # 7 joints + 1 F/T sensor weldjoint
    num_joints = 8
    arm_components = create_arm(
        arm_file_path=model_path.as_posix(), num_joints=num_joints, time_step=0.0
    )
    arm_plant_components = ArmPlantComponents(
        plant=arm_components.plant,
        plant_context=arm_components.plant.GetMyContextFromRoot(
            arm_components.diagram.CreateDefaultContext()
        ),
    )

    # Generate/ load joint data
    if traj_parameter_path is not None:
        is_fourier_series = os.path.exists(traj_parameter_path / "a_value.npy")
        if is_fourier_series:
            traj_attrs = FourierSeriesTrajectoryAttributes.load(
                traj_parameter_path, num_joints=num_joints
            )
            joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
                num_timesteps=num_data_points,
                time_horizon=time_horizon,
                traj_attrs=traj_attrs,
            )
        else:
            traj_attrs = BsplineTrajectoryAttributes.load(traj_parameter_path)
            traj = traj_attrs.to_bspline_trajectory()

            # Sample the trajectory
            q_numeric = np.empty((num_data_points, num_joints))
            q_dot_numeric = np.empty((num_data_points, num_joints))
            q_ddot_numeric = np.empty((num_data_points, num_joints))
            sample_times_s = np.linspace(
                traj.start_time(), traj.end_time(), num=num_data_points
            )
            for i, t in enumerate(sample_times_s):
                q_numeric[i] = traj.value(t).flatten()
                q_dot_numeric[i] = traj.EvalDerivative(t, derivative_order=1).flatten()
                q_ddot_numeric[i] = traj.EvalDerivative(t, derivative_order=2).flatten()

            joint_data = JointData(
                joint_positions=q_numeric,
                joint_velocities=q_dot_numeric,
                joint_accelerations=q_ddot_numeric,
                joint_torques=np.zeros_like(q_numeric),
                sample_times_s=sample_times_s,
            )
    else:
        joint_data = JointData.load_from_disk(joint_data_path)

    # Process joint data
    if do_process_joint_data:
        joint_data = process_joint_data(
            joint_data=joint_data,
            compute_velocities=compute_velocities,
            filter_positions=filter_positions,
            pos_filter_order=pos_filter_order,
            pos_cutoff_freq_hz=pos_cutoff_freq_hz,
            vel_filter_order=vel_filter_order,
            vel_cutoff_freq_hz=vel_cutoff_freq_hz,
            acc_filter_order=acc_filter_order,
            acc_cutoff_freq_hz=acc_cutoff_freq_hz,
            ft_sensor_force_order=ft_sensor_force_order,
            ft_sensor_force_cutoff_freq_hz=ft_sensor_force_cutoff_freq_hz,
            ft_sensor_torque_order=ft_sensor_torque_order,
            ft_sensor_torque_cutoff_freq_hz=ft_sensor_torque_cutoff_freq_hz,
        )
    else:
        assert not np.isnan(np.sum(joint_data.joint_accelerations)), (
            "NaNs found in joint_data.joint_accelerations. Consider setting "
            + "--process_joint_data."
        )

    # Generate data matrix
    A_data = construct_ft_data_matrix(
        plant_components=arm_plant_components,
        ft_body_name=sensor_body_name,
        ft_sensor_frame_name=sensor_frame_name,
        joint_data=joint_data,
    )

    if joint_data_path is None:
        logging.info(
            "Using model-predicted spatial forces as the F/T sensor measurements."
        )
        joint_data = compute_ft_sensor_measurements(
            arm_plant_components=arm_plant_components,
            joint_data=joint_data,
            ft_sensor_frame_name=sensor_frame_name,
        )
        ft_sensor_measurements = joint_data.ft_sensor_measurements
    else:
        # Use the measured F/T data
        ft_sensor_measurements = joint_data.ft_sensor_measurements
    ft_sensor_measurements = ft_sensor_measurements.flatten()

    (
        _,
        result,
        variable_names,
        variable_vec,
    ) = solve_ft_payload_sdp(
        ft_data_matrix=A_data,
        ft_data=ft_sensor_measurements,
    )
    if result.is_success():
        final_cost = result.get_optimal_cost()
        logging.info(f"Final cost: {final_cost}")
        wandb.log({"sdp_cost": final_cost})
        var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_vec)))
        logging.info(f"SDP result:\n{var_sol_dict}")

        if perform_eval:
            payload_mass = var_sol_dict["m(0)"]
            payload_hcom = np.array(
                [
                    var_sol_dict["hx(0)"],
                    var_sol_dict["hy(0)"],
                    var_sol_dict["hz(0)"],
                ]
            )
            payload_com = payload_hcom / payload_mass
            payload_rot_inertia = np.array(
                [
                    [
                        var_sol_dict["Ixx(0)"],
                        var_sol_dict["Ixy(0)"],
                        var_sol_dict["Ixz(0)"],
                    ],
                    [
                        var_sol_dict[f"Ixy(0)"],
                        var_sol_dict[f"Iyy(0)"],
                        var_sol_dict[f"Iyz(0)"],
                    ],
                    [
                        var_sol_dict["Ixz(0)"],
                        var_sol_dict["Iyz(0)"],
                        var_sol_dict["Izz(0)"],
                    ],
                ]
            )

            if payload_frame_name is None:
                logging.info("Payload parameters in the sensor frame:")
                logging.info(f"Payload mass: {payload_mass}")
                logging.info(f"Payload CoM: {payload_com}")
                logging.info(f"Payload inertia:\n{payload_rot_inertia}")
            else:
                # Transform into the payload frame
                last_link = arm_plant_components.plant.GetBodyByName(f"iiwa_link_7")
                payload_frame = arm_plant_components.plant.GetFrameByName(
                    payload_frame_name
                )
                plant_context = copy.deepcopy(arm_plant_components.plant_context)
                last_link.SetSpatialInertiaInBodyFrame(
                    plant_context,
                    SpatialInertia(
                        mass=payload_mass,
                        p_PScm_E=payload_com,
                        G_SP_E=UnitInertia(
                            Ixx=payload_rot_inertia[0, 0] / payload_mass,
                            Iyy=payload_rot_inertia[1, 1] / payload_mass,
                            Izz=payload_rot_inertia[2, 2] / payload_mass,
                            Ixy=payload_rot_inertia[0, 1] / payload_mass,
                            Ixz=payload_rot_inertia[0, 2] / payload_mass,
                            Iyz=payload_rot_inertia[1, 2] / payload_mass,
                        ),
                    ),
                )
                # Spatial inertia of payload about the payload frame origin,
                # expressed in the payload frame.
                M_PPayload_Payload = arm_plant_components.plant.CalcSpatialInertia(
                    context=plant_context,
                    frame_F=payload_frame,
                    body_indexes=[last_link.index()],
                )
                logging.info("Payload parameters in the specified payload frame:")
                logging.info(f"Payload mass: {M_PPayload_Payload.get_mass()}")
                logging.info(f"Payload CoM: {M_PPayload_Payload.get_com()}")
                logging.info(
                    "Payload inertia:\n"
                    + f"{M_PPayload_Payload.CalcRotationalInertia().CopyToFullMatrix3()}"
                )
    else:
        wandb.log({"sdp_cost": np.inf})
        logging.warning("Failed to solve inertial parameter SDP!")
        logging.info(f"Solution result:\n{result.get_solution_result()}")
        logging.info(f"Solver details:\n{result.get_solver_details()}")


if __name__ == "__main__":
    main()
