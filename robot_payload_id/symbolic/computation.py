from typing import List, Optional, Tuple

import numpy as np
import pydrake.symbolic as sym

from pydrake.all import MultibodyForces_
from tqdm import tqdm

from robot_payload_id.utils import (
    ArmPlantComponents,
    JointData,
    JointParameters,
    SymJointStateVariables,
)


def calculate_lumped_parameters(
    sym_arm_plant_components: ArmPlantComponents,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a symbolic factorization of the joint torques tau into an equivalent
    "data matrix", W, which depends only on the non-parameter variables, and a "lumped
    parameter vector", α, which depends only on parameters:
    W(n)*α(parameters) + w0(n) = 0.

    Args:
        sym_arm_plant_components (SymbolicArmPlantComponents): The symbolic plant and
        the associated components.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of (W(n), α(parameters),
        w0(n)).
    """
    forces = MultibodyForces_[sym.Expression](sym_arm_plant_components.plant)
    sym_arm_plant_components.plant.CalcForceElementsContribution(
        sym_arm_plant_components.plant_context, forces
    )
    sym_torques = sym_arm_plant_components.plant.CalcInverseDynamics(
        sym_arm_plant_components.plant_context,
        sym_arm_plant_components.state_variables.q_ddot.T,
        forces,
    )
    sym_parameters_arr = np.concatenate(
        [params.get_base_param_list() for params in sym_arm_plant_components.parameters]
    )
    W, alpha, w0 = sym.DecomposeLumpedParameters(sym_torques, sym_parameters_arr)
    return W, alpha, w0


def construct_data_matrix(
    W_sym: np.ndarray,
    alpha_sym: np.ndarray,
    w0_sym: np.ndarray,
    sym_state_variables: SymJointStateVariables,
    joint_data: JointData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructs a data matrix by evaluating the symbolic expressions using the joint
    data.

    Args:
        W_sym (np.ndarray): The symbolic data matrix of tau = W(n)*α(parameters) + w0(n).
        alpha_sym (np.ndarray): The symbolic lumped parameters of
        tau = W(n)*α(parameters) + w0(n).
        w0_sym (np.ndarray): The symbolic data vector of tau = W(n)*α(parameters) + w0(n).
        sym_state_variables (SymJointStateVariables): The symbolic joint state variables.
        joint_data (JointData): The joint data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of (W_data, W0_data, tau_data)
        that correspond to the evaluated variables in tau = W(n)*α(parameters) + w0(n).
    """
    num_joints = joint_data.joint_positions.shape[1]

    num_rows_per_joint = len(joint_data.sample_times_s) - 1
    num_rows = num_joints * num_rows_per_joint
    W_data = np.empty((num_rows, len(alpha_sym)))
    w0_data = np.empty((num_rows, 1))
    tau_data = np.empty((num_rows, 1))
    offset = 0
    for i in tqdm(range(num_rows_per_joint)):
        sym_env = {}
        for j in range(num_joints):
            sym_env[sym_state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_env[sym_state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_env[sym_state_variables.q_ddot[j]] = joint_data.joint_accelerations[
                i, j
            ]
            sym_env[sym_state_variables.tau[j]] = joint_data.joint_torques[i, j]

        W_data[offset : offset + num_joints, :] = sym.Evaluate(W_sym, sym_env)
        w0_data[offset : offset + num_joints] = sym.Evaluate(w0_sym, sym_env)
        tau_data[offset : offset + num_joints] = sym.Evaluate(
            sym_state_variables.tau, sym_env
        )

        offset += num_joints

    return W_data, w0_data, tau_data


def calc_lumped_parameters(
    sym_arm_plant_components: ArmPlantComponents,
    joint_data: JointData,
    gt_parameters: Optional[List[JointParameters]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtains the lumped parameters from the symbolic equations and computes their
    values using the joint data.

    Args:
        sym_arm_plant_components (SymbolicArmPlantComponents): The symbolic plant and
        associated components.
        joint_data (JointData): The robot joint data.
        gt_parameters (Optional[List[JointParameters]], optional): If not None, these
        GT parameters will be used for computing GT values of the lumped parameters.
    Returns:
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]: A tuple of (alpha_sym,
        alpha_estimated, alpha_gt) where alpha refers to the lumped parameters.
    """
    # w0 is just the symbolic form of tau such that W_sym + w0_sym = 0
    W_sym, alpha_sym, w0_sym = calculate_lumped_parameters(sym_arm_plant_components)
    print(f"Lumped parameters: {alpha_sym}")

    Wdata, _, tau_data = construct_data_matrix(
        W_sym=W_sym,
        alpha_sym=alpha_sym,
        w0_sym=w0_sym,
        sym_state_variables=sym_arm_plant_components.state_variables,
        joint_data=joint_data,
    )

    print(f"Condition number of Wdata: {np.linalg.cond(Wdata)}")
    print(f"Singular values of Wdata: {np.linalg.svd(Wdata, full_matrices=False)[1]}")

    alpha_estimated = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]

    alpha_gt = None
    if gt_parameters:
        sym_env = {}
        for sym_params, gt_params in zip(
            sym_arm_plant_components.parameters, gt_parameters
        ):
            sym_params_lst = sym_params.get_base_param_list()
            gt_params_lst = gt_params.get_base_param_list()
            assert len(sym_params_lst) == len(
                gt_params_lst
            ), "The number of GT parameters must equal the number of symbolic parameters!"
            for sym_param, gt_param in zip(sym_params_lst, gt_params_lst):
                sym_env[sym_param] = gt_param
            alpha_gt = sym.Evaluate(alpha_sym, sym_env)

    return alpha_sym, alpha_estimated, alpha_gt
