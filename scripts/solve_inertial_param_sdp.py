import argparse
import copy
import json
import logging
import os

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import sympy

from pydrake.all import (
    DecomposeAffineExpressions,
    MathematicalProgramResult,
    SpatialInertia,
    UnitInertia,
    from_sympy,
    to_sympy,
)
from scipy.linalg import lu

import wandb

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
    compute_base_param_mapping,
    extract_numeric_data_matrix_autodiff,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.eric_id.drake_torch_dynamics import (
    calc_inertia_entropic_divergence,
)
from robot_payload_id.optimization import solve_inertial_param_sdp
from robot_payload_id.utils import (
    ArmComponents,
    ArmPlantComponents,
    BsplineTrajectoryAttributes,
    FourierSeriesTrajectoryAttributes,
    JointData,
    get_plant_joint_params,
    inertia_to_pseudo_inertia,
    process_joint_data,
    pseudo_inertia_to_inertia,
    write_parameters_to_plant,
)
from robot_payload_id.utils.utils import compute_min_ellipsoid


def compute_entropic_divergence_to_gt_params(
    num_joints: int,
    arm_components: ArmComponents,
    var_sol_dict: Dict[str, float],
    payload_only: bool,
) -> None:
    """
    Compute and logs the inertial entropic divergence from the estimated parameters in
    `var_sol_dict` to the ground truth parameters in `arm_components.plant`.

    Zero entropic divergence means the estimated parameters are the same as the ground
    truth parameters. This is not possible as not all parameters are identifiable.
    """
    iterator = [num_joints - 1] if payload_only else range(num_joints)
    masses_estimated = np.array([var_sol_dict[f"m{i}(0)"] for i in iterator])
    coms_estimated = np.array(
        [
            [
                var_sol_dict[f"hx{i}(0)"],
                var_sol_dict[f"hy{i}(0)"],
                var_sol_dict[f"hz{i}(0)"],
            ]
            / var_sol_dict[f"m{i}(0)"]
            for i in iterator
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
            for i in iterator
        ]
    )

    joint_params_gt = get_plant_joint_params(
        plant=arm_components.plant,
        context=arm_components.plant.CreateDefaultContext(),
        add_rotor_inertia=False,
        add_reflected_inertia=False,
        add_viscous_friction=False,
        add_dynamic_dry_friction=False,
        payload_only=payload_only,
    )
    masses_gt = np.array([params.m for params in joint_params_gt])
    coms_gt = np.array([params.get_com() for params in joint_params_gt])
    rot_inertias_gt = np.array(
        [params.get_inertia_matrix() for params in joint_params_gt]
    )

    inertia_entropic_divergence = calc_inertia_entropic_divergence(
        masses_estimated,
        coms_estimated,
        rot_inertias_estimated,
        masses_gt,
        coms_gt,
        rot_inertias_gt,
    )
    logging.info(
        "Inertia entropic divergence from ground truth: "
        + f"{inertia_entropic_divergence}"
    )
    last_link_inertia_entropic_divergence = calc_inertia_entropic_divergence(
        masses_estimated[-1:],
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


def compute_base_parameter_errors(
    arm_components: ArmComponents,
    result: MathematicalProgramResult,
    identify_rotor_inertia: bool,
    identify_reflected_inertia: bool,
    identify_viscous_friction: bool,
    identify_dynamic_dry_friction: bool,
    payload_only: bool,
    base_param_mapping: np.ndarray,
    variable_vec: np.ndarray,
    base_variable_vec: np.ndarray,
    var_sol_dict: Dict[str, float],
    remove_term_threshold: float = 1e-6,
) -> None:
    # Get GT params
    inertial_params_gt = get_plant_joint_params(
        arm_components.plant,
        arm_components.plant.CreateDefaultContext(),
        add_rotor_inertia=identify_rotor_inertia,
        add_reflected_inertia=identify_reflected_inertia,
        add_viscous_friction=identify_viscous_friction,
        add_dynamic_dry_friction=identify_dynamic_dry_friction,
        payload_only=payload_only,
    )
    params_vec_gt = np.concatenate(
        [params.get_lumped_param_list() for params in inertial_params_gt]
    )
    # Assume that GT params are in the same order as variable_vec
    variable_names = [var.get_name() for var in variable_vec]
    var_gt_dict = dict(zip(variable_names, params_vec_gt))

    # Compute absolute base param errors
    if base_param_mapping is not None:
        base_params_vec_gt = params_vec_gt.T @ base_param_mapping
        base_variable_vec_numeric = np.array(
            [
                expression[0].Evaluate()
                for expression in result.GetSolution(base_variable_vec)
            ]
        )
    else:
        base_params_vec_gt = params_vec_gt
        base_variable_vec_numeric = result.GetSolution(variable_vec)
    base_param_abs_error = np.abs(base_params_vec_gt - base_variable_vec_numeric)
    logging.info(f"Total absolute base parameter error: {np.sum(base_param_abs_error)}")

    # Compute the identifiable parameters (approximate)
    # Remove terms with absolute value less than remove_term_threshold
    base_variable_vec_simplified = []
    for i, param in enumerate(base_variable_vec):
        memo = {}
        param_sympy = to_sympy(param, memo=memo)
        param_sympy_poly = sympy.Poly(param_sympy)
        to_remove = [
            abs(i) for i in param_sympy_poly.coeffs() if abs(i) < remove_term_threshold
        ]
        for i in to_remove:
            param_sympy_poly = param_sympy_poly.subs(i, 0)
        param_simplified = (
            from_sympy(param_sympy_poly, memo=memo)
            if not isinstance(param_sympy_poly, sympy.Poly)
            else param
        )
        base_variable_vec_simplified.append(param_simplified)
    A_matrix, _, x_vars = DecomposeAffineExpressions(base_variable_vec_simplified)
    U_matrix = lu(A_matrix)[2]
    # The basis columns correspond to the identifiable parameters
    basis_columns = {
        np.flatnonzero(U_matrix[i, :])[0] for i in range(U_matrix.shape[0])
    }
    identifiable_vars = [x_vars[i] for i in basis_columns]
    identifiable_vars_sorted = sorted(
        identifiable_vars, key=lambda x: int(x.get_name()[-4])
    )

    # Print GT and estimated values of identifiable parameters
    identifiable_vars_gt_values = [
        var_gt_dict[var.get_name()] for var in identifiable_vars_sorted
    ]
    identifiable_vars_estimated_values = [
        var_sol_dict[var.get_name()] for var in identifiable_vars_sorted
    ]
    logging.info(
        "Note that Drake changes the inertia frame when parsing SDFormat files. Hence, "
        + "the GT inertia values won't necessarily match the SDFormat file."
    )
    logging.info("Identifiable parameters:")
    for var, gt_value, estimated_value in zip(
        identifiable_vars_sorted,
        identifiable_vars_gt_values,
        identifiable_vars_estimated_values,
    ):
        rel_abs_error = (
            abs(gt_value - estimated_value) / abs(gt_value) if gt_value != 0 else np.nan
        )
        logging.info(
            f"{var.get_name():>25}: GT value: {gt_value:>12.6}, Estimated value: "
            + f"{estimated_value:>12.6}, Abs error: {abs(gt_value - estimated_value):>12.6} "
            + f", Rel abs error: {rel_abs_error:>12.6}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help="Use one link arm instead of 7-DOF arm.",
    )
    parser.add_argument(
        "--keep_unidentifiable_params",
        action="store_true",
        help="Keep and identify unidentifiable parameters instead of removing them.",
    )
    parser.add_argument(
        "--base_param_mapping_path",
        type=Path,
        help="Path to the base parameter mapping `.npy` file. Providing this will "
        + "override all default locations.",
    )
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
        "--identify_rotor_inertia",
        action="store_true",
        help="Identify rotor inertia. NOTE: It is recommended to identify reflected "
        + "inertia instead.",
    )
    parser.add_argument(
        "--not_identify_reflected_inertia",
        action="store_true",
        help="Do not identify reflected inertia.",
    )
    parser.add_argument(
        "--not_identify_viscous_friction",
        action="store_true",
        help="Do not identify viscous friction.",
    )
    parser.add_argument(
        "--not_identify_dynamic_dry_friction",
        action="store_true",
        help="Do not identify dynamic dry friction.",
    )
    parser.add_argument(
        "--use_euclidean_regularization",
        action="store_true",
        help="Use euclidean regularization instead of entropic divergence "
        + "regularization. Note that euclidian regularization is always used for the "
        + "non-inertial parameters.",
    )
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=1e-2,
        help="The regularization weight.",
    )
    parser.add_argument(
        "--perturb_scale",
        type=float,
        default=0.0,
        help="The scale of perturbation to apply to the ground truth parameters to get "
        + "a more realistic initial guess for simulation-based evaluation.",
    )
    parser.add_argument(
        "--payload_only",
        action="store_true",
        help="Only identify the 10 inertial parameters of the last link. All other "
        + "parameters are frozen. These are the parameters that we want to estimate "
        + "for payload identification.",
    )
    parser.add_argument(
        "--payload_frame_name",
        type=str,
        help="The frame to express the identified payload parameters in. This is only "
        + "used for logging when `--payload_only` is set. The frame must exist in "
        + "`--gt_model_path`.",
    )
    parser.add_argument(
        "--gt_model_path",
        type=Path,
        help="Path to the ground truth robot model. This could for example have a "
        + "payload attached. If not provided, the ground truth model is assumed to be "
        + "the same as the model used for identification (e.g. vanilla iiwa).",
    )
    parser.add_argument(
        "--initial_param_path",
        type=Path,
        help="Path to the initial parameter `.npy` file. If not provided, the initial "
        + "parameters are set to the model's parameters.",
    )
    parser.add_argument(
        "--output_param_path",
        type=Path,
        help="Path to the output parameter `.npy` file. If not provided, the parameters "
        + "are not saved to disk.",
    )
    parser.add_argument(
        "--use_initial_params_for_regularization",
        action="store_true",
        help="Use the initial parameters for regularization. This might not be a good "
        + "idea if the initial parameters contain random values which can occur when "
        + "some of the parameters are unidentifiable.",
    )
    parser.add_argument(
        "--known_max_mass",
        type=float,
        help="The known maximum mass of the robot. This is used to set the upper bound "
        + "for the combined mass parameters.",
    )
    parser.add_argument(
        "--use_commanded_torques",
        action="store_true",
        help="Whether to use the commanded torques instead of the measured torques in "
        + "joint data. This can be useful if we want to use the identified parameters "
        + "for control. Note that the commanded and measured torques can be quite "
        + "different.",
    )
    parser.add_argument(
        "--process_joint_data",
        action="store_true",
        help="Whether to process the joint data before using it.",
    )
    parser.add_argument(
        "--num_endpoints_to_remove",
        type=int,
        default=1,
        help="The number of endpoints to remove from the beginning and end of the "
        + "trajectory. This is useful as the sample times are not always increasing "
        + "with the same period at the beginning and end of the trajectory. Only used "
        + "if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--not_compute_velocities",
        action="store_true",
        help="Whether take the velocities in `joint_data` instead of computing them "
        + "from the positions. Only used if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--not_filter_positions",
        action="store_true",
        help="Whether to not filter the joint positions. Only used if `--process_joint_data` "
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
        "--torque_order",
        type=int,
        default=10,
        help="The order of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_cutoff_freq_hz",
        type=float,
        default=10.0,
        help="The cutoff frequency of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["disabled", "online", "offline"],
        default="disabled",
        help="The mode to use for Weights & Biases logging.",
    )
    parser.add_argument(
        "--payload_mesh_path",
        type=str,
        default=None,
        help="Path to the payload mesh file. This is used to add the bounding "
        "ellipsoid constraint for payload identification. No constraint is added if "
        "not provided.",
    )
    parser.add_argument(
        "--payload_mesh_pose",
        type=json.loads,
        default=None,
        help="The homogenous rigid transform matrix from the last link frame to the "
        "mesh frame, expressed in the last link's frame. Required and only used if "
        "`payload_mesh_path` is provided.",
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
    base_param_mapping_path = args.base_param_mapping_path
    traj_parameter_path = args.traj_parameter_path
    joint_data_path = args.joint_data_path
    num_data_points = args.num_data_points
    time_horizon = args.time_horizon
    identify_rotor_inertia = args.identify_rotor_inertia
    identify_reflected_inertia = not args.not_identify_reflected_inertia
    identify_viscous_friction = not args.not_identify_viscous_friction
    identify_dynamic_dry_friction = not args.not_identify_dynamic_dry_friction
    use_euclidean_regularization = args.use_euclidean_regularization
    regularization_weight = args.regularization_weight
    perturb_scale = args.perturb_scale
    payload_only = args.payload_only
    payload_frame_name = args.payload_frame_name
    gt_model_path = args.gt_model_path
    initial_param_path = args.initial_param_path
    output_param_path = args.output_param_path
    use_initial_params_for_regularization = args.use_initial_params_for_regularization
    known_max_mass = args.known_max_mass
    use_commanded_torques = args.use_commanded_torques
    do_process_joint_data = args.process_joint_data
    num_endpoints_to_remove = args.num_endpoints_to_remove
    compute_velocities = not args.not_compute_velocities
    filter_positions = not args.not_filter_positions
    pos_filter_order = args.pos_order
    pos_cutoff_freq_hz = args.pos_cutoff_freq_hz
    vel_filter_order = args.vel_order
    vel_cutoff_freq_hz = args.vel_cutoff_freq_hz
    acc_filter_order = args.acc_order
    acc_cutoff_freq_hz = args.acc_cutoff_freq_hz
    torque_filter_order = args.torque_order
    torque_cutoff_freq_hz = args.torque_cutoff_freq_hz
    wandb_mode = args.wandb_mode
    payload_mesh_path = args.payload_mesh_path
    payload_mesh_pose = args.payload_mesh_pose
    perform_eval = not args.not_perform_eval

    assert (traj_parameter_path is not None) != (joint_data_path is not None), (
        "One but not both of `--traj_parameter_path` and `--joint_data_path` should be "
        + "set."
    )

    if payload_only and regularization_weight > 0.0:
        logging.warning(
            "Regularization is not recommended when identifying only the payload "
            + "parameters as these are fully identifiable."
        )

    if payload_mesh_path is not None:
        assert payload_mesh_pose is not None, (
            "The payload_mesh_pose must be provided if the payload_mesh_path is "
            + "provided."
        )

    if payload_mesh_pose is not None:
        payload_mesh_pose = np.array(payload_mesh_pose)
        assert payload_mesh_pose.shape == (
            4,
            4,
        ), "The payload_mesh_pose must be a 4x4 matrix."

    logging.basicConfig(level=args.log_level)
    wandb.init(
        project="robot_payload_id",
        name=f"inertial_param_sdp ({datetime.now()})",
        config=vars(args),
        mode=wandb_mode,
    )

    # Create arm
    num_joints = 1 if args.use_one_link_arm else 7
    # NOTE: This model must not have a payload attached. Otherwise, the w0 term will be
    # wrong and include the payload inertia.
    model_path = (
        "./models/one_link_arm.sdf"
        if args.use_one_link_arm
        else "./models/iiwa.dmd.yaml"
    )
    arm_components = create_arm(
        arm_file_path=model_path, num_joints=num_joints, time_step=0.0
    )

    # Load parameters
    if initial_param_path is not None:
        logging.info(f"Loading initial parameters from {initial_param_path}")
        var_sol_dict = np.load(initial_param_path, allow_pickle=True).item()
        arm_plant_components = write_parameters_to_plant(arm_components, var_sol_dict)
    else:
        arm_plant_components = ArmPlantComponents(
            plant=arm_components.plant,
            plant_context=arm_components.plant.CreateDefaultContext(),
        )

    # Model for computing GT values
    gt_model_path = str(gt_model_path) if gt_model_path is not None else model_path
    arm_components_gt = create_arm(
        arm_file_path=gt_model_path, num_joints=num_joints, time_step=0.0
    )
    arm_plant_components_gt = ArmPlantComponents(
        plant=arm_components_gt.plant,
        plant_context=arm_components_gt.plant.CreateDefaultContext(),
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
        if use_commanded_torques:
            joint_data.joint_torques = np.load(
                joint_data_path / "commanded_joint_torques.npy"
            )

    # Process joint data
    if do_process_joint_data:
        joint_data = process_joint_data(
            joint_data=joint_data,
            num_endpoints_to_remove=num_endpoints_to_remove,
            compute_velocities=compute_velocities,
            filter_positions=filter_positions,
            pos_filter_order=pos_filter_order,
            pos_cutoff_freq_hz=pos_cutoff_freq_hz,
            vel_filter_order=vel_filter_order,
            vel_cutoff_freq_hz=vel_cutoff_freq_hz,
            acc_filter_order=acc_filter_order,
            acc_cutoff_freq_hz=acc_cutoff_freq_hz,
            torque_filter_order=torque_filter_order,
            torque_cutoff_freq_hz=torque_cutoff_freq_hz,
        )
    else:
        assert not np.isnan(np.sum(joint_data.joint_accelerations)), (
            "NaNs found in joint_data.joint_accelerations. Consider setting "
            + "--process_joint_data."
        )

    # Generate data matrix
    (
        W_data_raw,
        w0_data,
        _,
    ) = extract_numeric_data_matrix_autodiff(
        plant_components=arm_plant_components,
        joint_data=joint_data,
        add_rotor_inertia=identify_rotor_inertia,
        add_reflected_inertia=identify_reflected_inertia,
        add_viscous_friction=identify_viscous_friction,
        add_dynamic_dry_friction=identify_dynamic_dry_friction,
        payload_only=payload_only,
    )
    if joint_data_path is None:
        # Use the model-predicted torques
        logging.info("Using model-predicted torques as the measured torques.")
        (
            _,
            _,
            tau_data,
        ) = extract_numeric_data_matrix_autodiff(
            plant_components=arm_plant_components_gt,
            joint_data=joint_data,
            add_rotor_inertia=identify_rotor_inertia,
            add_reflected_inertia=identify_reflected_inertia,
            add_viscous_friction=identify_viscous_friction,
            add_dynamic_dry_friction=identify_dynamic_dry_friction,
            payload_only=payload_only,
        )
    else:
        # Use the measured torques
        tau_data = joint_data.joint_torques.flatten()
    # Transform from affine `tau = W * params + w0` into linear `(tau - w0) = W * params`
    tau_data -= w0_data

    # All parameters are identifiable when only identifying the payload parameters
    if not args.keep_unidentifiable_params and not payload_only:
        # Load base parameter mapping
        base_param_mapping = None
        if base_param_mapping_path is not None:
            logging.info(
                f"Loading base parameter mapping from {base_param_mapping_path}"
            )
            base_param_mapping = np.load(base_param_mapping_path)
        elif traj_parameter_path is not None:
            base_param_mapping_path1 = traj_parameter_path / "base_param_mapping.npy"
            base_param_mapping_path2 = traj_parameter_path / "../base_param_mapping.npy"
            if os.path.exists(base_param_mapping_path1):
                logging.info(
                    f"Loading base parameter mapping from {base_param_mapping_path1}"
                )
                base_param_mapping = np.load(base_param_mapping_path1)
            elif os.path.exists(base_param_mapping_path2):
                logging.info(
                    f"Loading base parameter mapping from {base_param_mapping_path2}"
                )
                base_param_mapping = np.load(base_param_mapping_path2)

        # Recompute base parameter mapping if it has wrong shape or is not provided
        if (
            base_param_mapping is None
            or W_data_raw.shape[1] != base_param_mapping.shape[0]
        ):
            logging.warning(
                "Base parameter mapping not provided or has wrong shape! Recomputing "
                + "the base parameter mapping..."
            )
            if num_data_points > 2000:
                logging.warning(
                    "Number of data points is large. Using 2000 random data points "
                    + "to compute the base parameter mapping."
                )
                num_random_points = 2000
                joint_data_random = JointData(
                    joint_positions=np.random.rand(num_random_points, num_joints) - 0.5,
                    joint_velocities=np.random.rand(num_random_points, num_joints)
                    - 0.5,
                    joint_accelerations=np.random.rand(num_random_points, num_joints)
                    - 0.5,
                    joint_torques=np.zeros((num_random_points, num_joints)),
                    sample_times_s=np.zeros(num_random_points),
                )
                W_data_random, _, _ = extract_numeric_data_matrix_autodiff(
                    plant_components=arm_plant_components,
                    joint_data=joint_data_random,
                    add_rotor_inertia=identify_rotor_inertia,
                    add_reflected_inertia=identify_reflected_inertia,
                    add_viscous_friction=identify_viscous_friction,
                    add_dynamic_dry_friction=identify_dynamic_dry_friction,
                    payload_only=payload_only,
                )
            else:
                W_data_random = W_data_raw
            base_param_mapping = compute_base_param_mapping(W_data_random)

        logging.info(
            f"{base_param_mapping.shape[1]} out of {base_param_mapping.shape[0]} "
            + "parameters are identifiable."
        )

        if base_param_mapping.shape[0] == base_param_mapping.shape[1]:
            logging.warning(
                "All parameters are identifiable. Not applying SVD projection."
            )
            base_param_mapping = None
    else:
        base_param_mapping = None

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

    # Construct initial parameter guess
    if use_initial_params_for_regularization and initial_param_path is not None:
        logging.info("Using loaded parameters for regularization.")
        params_guess = get_plant_joint_params(
            arm_plant_components.plant,
            arm_plant_components.plant_context,
            add_rotor_inertia=identify_rotor_inertia,
            add_reflected_inertia=identify_reflected_inertia,
            add_viscous_friction=identify_viscous_friction,
            add_dynamic_dry_friction=identify_dynamic_dry_friction,
            payload_only=payload_only,
        )
    else:
        params_guess = get_plant_joint_params(
            arm_components_gt.plant,
            arm_components_gt.plant.CreateDefaultContext(),
            add_rotor_inertia=identify_rotor_inertia,
            add_reflected_inertia=identify_reflected_inertia,
            add_viscous_friction=identify_viscous_friction,
            add_dynamic_dry_friction=identify_dynamic_dry_friction,
            payload_only=payload_only,
        )
        if perturb_scale > 0.0:
            for params in params_guess:
                params.perturb(perturb_scale)

    inital_last_link_params = (
        get_plant_joint_params(
            arm_plant_components.plant,
            arm_plant_components.plant_context,
            add_rotor_inertia=identify_rotor_inertia,
            add_reflected_inertia=identify_reflected_inertia,
            add_viscous_friction=identify_viscous_friction,
            add_dynamic_dry_friction=identify_dynamic_dry_friction,
            payload_only=payload_only,
        )[-1]
        if payload_only
        else None
    )

    if payload_only and payload_mesh_path is not None:
        # Compute the bounding ellipsoid in the payload obejct's frame
        payload_bounding_ellipsoid = compute_min_ellipsoid(
            payload_mesh_path, transform=payload_mesh_pose
        )
    else:
        payload_bounding_ellipsoid = None

    (
        _,
        result,
        variable_names,
        variable_vec,
        base_variable_vec,
    ) = solve_inertial_param_sdp(
        num_links=num_joints,
        W_data=W_data,
        tau_data=tau_data,
        base_param_mapping=base_param_mapping,
        regularization_weight=regularization_weight,
        params_guess=params_guess,
        use_euclidean_regularization=use_euclidean_regularization,
        identify_rotor_inertia=identify_rotor_inertia,
        identify_reflected_inertia=identify_reflected_inertia,
        identify_viscous_friction=identify_viscous_friction,
        identify_dynamic_dry_friction=identify_dynamic_dry_friction,
        payload_only=payload_only,
        known_max_mass=known_max_mass,
        initial_last_link_params=inital_last_link_params,
        payload_bounding_ellipsoid=payload_bounding_ellipsoid,
        solver_kPrintToConsole=args.kPrintToConsole,
    )
    if result.is_success():
        final_cost = result.get_optimal_cost()
        logging.info(f"Final cost: {final_cost}")
        wandb.log({"sdp_cost": final_cost})
        var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_vec)))
        logging.info(f"SDP result:\n{var_sol_dict}")

        if perform_eval:
            compute_entropic_divergence_to_gt_params(
                num_joints=num_joints,
                arm_components=arm_components_gt,
                var_sol_dict=var_sol_dict,
                payload_only=payload_only,
            )
            compute_base_parameter_errors(
                arm_components=arm_components_gt,
                result=result,
                identify_rotor_inertia=identify_rotor_inertia,
                identify_reflected_inertia=identify_reflected_inertia,
                identify_viscous_friction=identify_viscous_friction,
                identify_dynamic_dry_friction=identify_dynamic_dry_friction,
                payload_only=payload_only,
                base_param_mapping=base_param_mapping,
                variable_vec=variable_vec,
                base_variable_vec=base_variable_vec,
                var_sol_dict=var_sol_dict,
            )

            if payload_only:
                # Compute the difference in the last link's parameters. This corresponds
                # to the payload parameters if `initial_param_path` are the parameters
                # without payload.

                # We can subtract the lumped parameters as they are all in the last
                # link's frame.
                last_link_idx = num_joints - 1
                payload_mass = (
                    var_sol_dict[f"m{last_link_idx}(0)"] - inital_last_link_params.m
                )
                payload_hcom = (
                    np.array(
                        [
                            var_sol_dict[f"hx{last_link_idx}(0)"],
                            var_sol_dict[f"hy{last_link_idx}(0)"],
                            var_sol_dict[f"hz{last_link_idx}(0)"],
                        ]
                    )
                    - inital_last_link_params.m * inital_last_link_params.get_com()
                )
                payload_com = payload_hcom / payload_mass
                payload_rot_inertia = (
                    np.array(
                        [
                            [
                                var_sol_dict[f"Ixx{last_link_idx}(0)"],
                                var_sol_dict[f"Ixy{last_link_idx}(0)"],
                                var_sol_dict[f"Ixz{last_link_idx}(0)"],
                            ],
                            [
                                var_sol_dict[f"Ixy{last_link_idx}(0)"],
                                var_sol_dict[f"Iyy{last_link_idx}(0)"],
                                var_sol_dict[f"Iyz{last_link_idx}(0)"],
                            ],
                            [
                                var_sol_dict[f"Ixz{last_link_idx}(0)"],
                                var_sol_dict[f"Iyz{last_link_idx}(0)"],
                                var_sol_dict[f"Izz{last_link_idx}(0)"],
                            ],
                        ]
                    )
                    - inital_last_link_params.get_inertia_matrix()
                )

                if payload_frame_name is None:
                    logging.info(
                        "Difference in the last link's parameters (payload parameters). "
                        + "Note that these are in the last link's frame:"
                    )
                    logging.info(f"Payload mass: {payload_mass}")
                    logging.info(f"Payload CoM: {payload_com}")
                    logging.info(f"Payload inertia:\n{payload_rot_inertia}")
                else:
                    # Transform into the payload frame
                    last_link = arm_plant_components_gt.plant.GetBodyByName(
                        f"iiwa_link_7"
                    )
                    payload_frame = arm_plant_components_gt.plant.GetFrameByName(
                        payload_frame_name
                    )
                    plant_context = copy.deepcopy(arm_plant_components_gt.plant_context)
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
                    M_PPayload_Payload = (
                        arm_plant_components_gt.plant.CalcSpatialInertia(
                            context=plant_context,
                            frame_F=payload_frame,
                            body_indexes=[last_link.index()],
                        )
                    )
                    # Express inerita about CoM to match SDFormat convention
                    M_PPayloadcom_Payload = M_PPayload_Payload.Shift(
                        M_PPayload_Payload.get_com()
                    )
                    logging.info(
                        "Difference in the last link's parameters (payload parameters). "
                        + "Note that these are in specified payload frame:"
                    )
                    payload_mass = M_PPayloadcom_Payload.get_mass()
                    logging.info(f"Payload mass: {payload_mass}")
                    com_PPayload_Payload = M_PPayload_Payload.get_com()
                    logging.info(f"Payload CoM: {com_PPayload_Payload}")
                    I_PPayloadcom_Payload = (
                        M_PPayloadcom_Payload.CalcRotationalInertia().CopyToFullMatrix3()
                    )
                    logging.info(
                        "Payload inertia (about CoM):\n" + f"{I_PPayloadcom_Payload}\n"
                    )

                    # Pseudo inertia of payload about the payload frame origin,
                    # expressed in the payload frame.
                    # pseudo_inertia = inertia_to_pseudo_inertia(
                    #     m=payload_mass,
                    #     com=com_PPayload_Payload,
                    #     inertia=I_PPayload_Payload,
                    # )
                    # logging.info(f"Pseudo inertia of payload:\n{pseudo_inertia}")

        if output_param_path is not None:
            logging.info(f"Saving parameters to {output_param_path}")
            directory: Path = output_param_path.parent
            directory.mkdir(parents=True, exist_ok=True)
            np.save(output_param_path, var_sol_dict)
    else:
        wandb.log({"sdp_cost": np.inf})
        logging.warning("Failed to solve inertial parameter SDP!")
        logging.info(f"Solution result:\n{result.get_solution_result()}")
        logging.info(f"Solver details:\n{result.get_solver_details()}")


if __name__ == "__main__":
    main()
