"""
Methods for computing the numeric form of the inertial parameter least-squares data
matrix.
"""

import logging
import time

from datetime import timedelta
from typing import Optional, Tuple, Union

import numpy as np

from pydrake.all import (
    AutoDiffXd,
    DecomposeLumpedParameters,
    Evaluate,
    Expression,
    MathematicalProgram,
    MultibodyForces_,
)
from tqdm import tqdm

from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_autodiff_plant, create_symbolic_plant
from robot_payload_id.utils import ArmComponents, ArmPlantComponents, JointData


def extract_numeric_data_matrix_symbolic(
    joint_data: JointData,
    prog: Optional[MathematicalProgram] = None,
    use_implicit_dynamics: bool = False,
    use_w0: bool = False,
    use_one_link_arm: bool = False,
):
    """Uses symbolic to extract the numeric data matrix. This does not scale to a large
    number of links such as a 7-DOF arm.
    NOTE: This method is currently experimental and should be used with care.
    TODO: Clean up + add documentation.
    """
    assert not use_w0 or use_implicit_dynamics, "Can only use w0 with implicit dynamics"

    urdf_path = (
        "./models/one_link_arm.urdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )
    num_joints = 1 if use_one_link_arm else 7
    time_step = 0 if use_implicit_dynamics else 1e-3

    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=time_step
    )
    sym_plant_components = create_symbolic_plant(
        arm_components=arm_components, prog=prog
    )

    sym_parameters_arr = np.concatenate(
        [params.get_base_param_list() for params in sym_plant_components.parameters]
    )
    if use_implicit_dynamics:
        derivatives = (
            sym_plant_components.plant_context.Clone().get_mutable_continuous_state()
        )
        derivatives.SetFromVector(
            np.hstack(
                (
                    0 * sym_plant_components.state_variables.q_dot,
                    sym_plant_components.state_variables.q_ddot,
                )
            )
        )
        residual = sym_plant_components.plant.CalcImplicitTimeDerivativesResidual(
            sym_plant_components.plant_context, derivatives
        )
        W_sym, alpha_sym, w0_sym = DecomposeLumpedParameters(
            residual[int(len(residual) / 2) :], sym_parameters_arr
        )
    else:
        forces = MultibodyForces_[Expression](sym_plant_components.plant)
        sym_plant_components.plant.CalcForceElementsContribution(
            sym_plant_components.plant_context, forces
        )
        sym_torques = sym_plant_components.plant.CalcInverseDynamics(
            sym_plant_components.plant_context,
            sym_plant_components.state_variables.q_ddot.T,
            forces,
        )
        W_sym, alpha_sym, w0_sym = DecomposeLumpedParameters(
            sym_torques, sym_parameters_arr
        )

    # Substitute data values
    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * (3 if use_one_link_arm else 10)
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    w0_data = np.zeros(num_timesteps * num_joints)
    tau_data = joint_data.joint_torques.flatten()

    state_variables = sym_plant_components.state_variables
    for i in tqdm(range(num_timesteps)):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_to_val[state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_to_val[state_variables.q_ddot[j]] = joint_data.joint_accelerations[i, j]
            if use_implicit_dynamics:
                sym_to_val[state_variables.tau[j]] = joint_data.joint_torques[i, j]
        W_data[i * num_joints : (i + 1) * num_joints, :] = Evaluate(W_sym, sym_to_val)
        if use_w0:
            w0_data[i * num_joints : (i + 1) * num_joints] = Evaluate(
                w0_sym, sym_to_val
            )

    if use_w0:
        return W_data, alpha_sym, w0_data
    return W_data, alpha_sym, tau_data


def extract_numeric_data_matrix_through_symbolic_decomposition_with_dynamic_substitution(
    symbolic_plant_components: ArmPlantComponents, joint_data: JointData
):
    """Algorithm 1 from Andy's thesis for extracting the numeric data matrix using
    alternation between symbolic and numerica substitution. Scales to a large number of
    links.
    NOTE: This method is currently experimental and should be used with care.
    TODO: Clean up + add documentation.
    """
    num_timesteps = len(joint_data.sample_times_s)
    num_links = symbolic_plant_components.state_variables.q.shape[-1]
    num_lumped_params = (
        51  # num_links * 10  # TODO: True in theory but maybe not in practice
    )
    W_data = np.zeros((num_timesteps * num_links, num_lumped_params))
    tau_data = joint_data.joint_torques.flatten()

    plant = symbolic_plant_components.plant
    for i in tqdm(range(num_timesteps)):
        # Substitute symbolic variables with numeric values
        new_context = symbolic_plant_components.plant_context.Clone()
        plant.SetPositions(new_context, joint_data.joint_positions[i])
        plant.SetVelocities(new_context, joint_data.joint_velocities[i])

        # Calculate inverse dynamics
        forces = MultibodyForces_[Expression](plant)
        plant.CalcForceElementsContribution(new_context, forces)
        sym_torques = plant.CalcInverseDynamics(
            new_context, joint_data.joint_accelerations[i], forces
        )

        # Decompose symbolic expressions
        sym_parameters_arr = np.concatenate(
            [
                params.get_base_param_list()
                for params in symbolic_plant_components.parameters
            ]
        )
        W, alpha, w0 = DecomposeLumpedParameters(sym_torques, sym_parameters_arr)

        try:
            W_data[i * num_links : (i + 1) * num_links, :] = Evaluate(W, {})
            alpha_sym = alpha
        except:
            logging.warning("Substitution failed:")
            logging.warning("W_data shape:", W_data.shape)
            logging.warning("W shape:", W.shape)
            logging.warning("i:", i)
            logging.warning("------------------")
            continue

    return W_data, alpha_sym, tau_data


def extract_numeric_data_matrix_autodiff(
    arm_components: Union[ArmComponents, ArmPlantComponents],
    joint_data: JointData,
    add_rotor_inertia: bool,
    add_viscous_friction: bool,
    add_dynamic_dry_friction: bool,
    use_progress_bar: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts the numeric data matrix using autodiff. This scales to a large number of
    links.

    Args:
        arm_components (Union[ArmComponents, ArmPlantComponents]): The arm components
            or the autodiff arm plant components.
        joint_data (JointData): The joint data. Only the joint positions, velocities,
            and accelerations are used for the data matrix computation. The joint
            torques are flattened and returned.
        add_rotor_inertia (bool): Whether to add rotor inertia as a parameter.
        add_viscous_friction (bool): Whether to add viscous friction as a parameter.
        add_dynamic_dry_friction (bool): Whether to add dynamic dry friction as a
            parameter.
        use_progress_bar (bool, optional): Whether to use a progress bar.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the numeric data matrix and
            the numeric torque data. The numeric data matrix has shape (num_joints *
            num_timesteps, num_lumped_params) and the numeric torque data has shape
            (num_joints * num_timesteps,).
    """
    # Create autodiff arm
    ad_plant_components = (
        arm_components
        if isinstance(arm_components, ArmPlantComponents)
        else create_autodiff_plant(
            arm_components=arm_components,
            add_rotor_inertia=add_rotor_inertia,
            add_viscous_friction=add_viscous_friction,
            add_dynamic_dry_friction=add_dynamic_dry_friction,
        )
    )
    if add_viscous_friction:
        viscous_frictions_ad = np.array(
            [params.viscous_friction for params in ad_plant_components.parameters]
        )
    if add_dynamic_dry_friction:
        dynamic_dry_frictions_ad = np.array(
            [params.dynamic_dry_friction for params in ad_plant_components.parameters]
        )

    # Extract data matrix
    num_joints = joint_data.joint_positions.shape[1]
    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = sum(
        [
            len(params.get_lumped_param_list())
            for params in ad_plant_components.parameters
        ]
    )
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    tau_data = joint_data.joint_torques.flatten()

    for i in tqdm(
        range(num_timesteps),
        desc="Extracting data matrix",
        disable=not use_progress_bar,
    ):
        # Set joint data
        ad_plant_components.plant.SetPositions(
            ad_plant_components.plant_context, joint_data.joint_positions[i]
        )
        ad_plant_components.plant.SetVelocities(
            ad_plant_components.plant_context, joint_data.joint_velocities[i]
        )

        # Compute inverse dynamics
        forces = MultibodyForces_[AutoDiffXd](ad_plant_components.plant)
        ad_plant_components.plant.CalcForceElementsContribution(
            ad_plant_components.plant_context, forces
        )
        ad_torques = ad_plant_components.plant.CalcInverseDynamics(
            ad_plant_components.plant_context,
            joint_data.joint_accelerations[i],
            forces,
        )

        if add_viscous_friction:
            ad_torques += viscous_frictions_ad * joint_data.joint_velocities[i]
        if add_dynamic_dry_friction:
            ad_torques += dynamic_dry_frictions_ad * np.sign(
                joint_data.joint_velocities[i]
            )

        # Differentiate w.r.t. parameters
        ad_torques_derivative = np.vstack(
            [joint_torques.derivatives() for joint_torques in ad_torques]
        )
        W_data[i * num_joints : (i + 1) * num_joints, :] = ad_torques_derivative

    return W_data, tau_data


def compute_base_param_mapping(
    W_data: np.ndarray, scale_by_singular_values: bool = False, tol: float = 1e-6
) -> np.ndarray:
    """Computes the base parameter mapping matrix that maps the full parameters to the
    identifiable base parameters. It corresponds to the part of V in the SVD that
    corresponds to non-zero singular values.

    Args:
        W_data (np.ndarray): The data matrix of shape (num_joints * num_timesteps,
            num_lumped_params). This should be a random numeric data matrix that
            excites all the parameters. Parameters that are not excited will not be
            included in the base parameters.
        scale_by_singular_values (bool): Whether to scale the base parameter mapping by
            the reciprocals of the singular values.
        tol (float): The tolerance for considering singular values as non-zero.

    Returns:
        np.ndarray: The base parameter mapping matrix.
    """
    # NOTE: This might lead to running out of memory for large matrices. W_data
    # is sparse and hence it might be possible to use a sparse SVD. However,
    # this would make reconstruction difficutl.
    logging.info(
        "Computing SVD for base parameter mapping. This might take a while for "
        + "large data matrices."
    )
    svd_start = time.time()
    _, S, VT = np.linalg.svd(W_data)
    logging.info(f"SVD took {timedelta(seconds=time.time() - svd_start)}")
    V = VT.T
    mask = np.abs(S) > tol
    base_param_mapping = V[:, mask]
    if scale_by_singular_values:
        base_param_mapping *= 1 / S[mask]
    return base_param_mapping
