import time

from datetime import timedelta
from typing import List, Optional

import numpy as np
import sympy

from pydrake.all import (
    Evaluate,
    Expression,
    MathematicalProgram,
    MultibodyForces_,
    from_sympy,
    to_sympy,
)
from tqdm import tqdm

from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_symbolic_plant
from robot_payload_id.utils import JointData


def get_data(plant, num_q, N=100, add_noise: bool = False) -> JointData:
    dim_q = (N, num_q)
    q = 1.0 * np.random.uniform(size=dim_q)
    v = 3.0 * np.random.uniform(size=dim_q)
    vd = 5.0 * np.random.uniform(size=dim_q)
    tau_gt = np.empty(dim_q)

    context = plant.CreateDefaultContext()
    for i, (q_curr, v_curr, v_dot_curr) in enumerate(zip(q, v, vd)):
        plant.SetPositions(context, q_curr)
        plant.SetVelocities(context, v_curr)
        forces = MultibodyForces_(plant)
        plant.CalcForceElementsContribution(context, forces)
        tau_gt[i] = plant.CalcInverseDynamics(context, v_dot_curr, forces)

    if add_noise:
        q += 0.001 * np.random.normal(size=dim_q)
        v += 0.01 * np.random.normal(size=dim_q)
        vd += 0.05 * np.random.normal(size=dim_q)

    joint_data = JointData(
        joint_positions=q,
        joint_velocities=v,
        joint_accelerations=vd,
        joint_torques=tau_gt,
        sample_times_s=np.arange(N) * 1e-3,
    )
    return joint_data


def extract_data_matrix_symbolic(
    prog: Optional[MathematicalProgram] = None,
    use_implicit_dynamics: bool = False,
    use_w0: bool = False,
    use_one_link_arm: bool = False,
):
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
        arm_components=arm_components, prog=prog, use_lumped_parameters=True
    )

    sym_parameters_arr = np.concatenate(
        [params.get_lumped_param_list() for params in sym_plant_components.parameters]
    )

    forces = MultibodyForces_[Expression](sym_plant_components.plant)
    sym_plant_components.plant.CalcForceElementsContribution(
        sym_plant_components.plant_context, forces
    )
    sym_torques = sym_plant_components.plant.CalcInverseDynamics(
        sym_plant_components.plant_context,
        sym_plant_components.state_variables.q_ddot.T,
        forces,
    )

    # NOTE: Could parallelise this if needed
    W_sym: List[Expression] = []
    expression: Expression
    start_time = time.time()
    for expression in tqdm(sym_torques, desc="Computing W_sym"):
        # Simplification not needed if using lumped parameters in Drake's inverse\
        # dynamics computation
        # memo = {}
        # expression_sympy = to_sympy(expression, memo=memo)
        # print("Start simplify")
        # start_simplify = time.time()
        # # Cancel should be sufficient here
        # # simplified_expression_sympy = sympy.simplify(expression_sympy)
        # simplified_expression_sympy = sympy.cancel(expression_sympy)
        # print(f"Simplification took {time.time() - start_simplify} seconds")
        # simplified_expression: Expression = from_sympy(
        #     simplified_expression_sympy, memo=memo
        # )
        # W_sym.append(simplified_expression.Jacobian(sym_parameters_arr))

        W_sym.append(expression.Jacobian(sym_parameters_arr))
    W_sym = np.vstack(W_sym)
    print("Time to compute W_sym:", timedelta(seconds=time.time() - start_time))

    # print("W_sym:\n", W_sym)

    start_time = time.time()
    for i, row in tqdm(enumerate(W_sym), total=len(W_sym), desc="Saving to disk (row)"):
        for j, expression in tqdm(
            enumerate(row), total=len(row), desc="    Saving to disk (column)"
        ):
            memo = {}
            expression_sympy = to_sympy(expression, memo=memo)
            np.save(f"W_{i}_{j}_sympy.npy", expression_sympy)
    print("Time to save to disk:", timedelta(seconds=time.time() - start_time))

    # Substitute data values and compute least squares fit
    joint_data = get_data(num_q=num_joints, plant=arm_components.plant)
    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * 10
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    tau_data = joint_data.joint_torques.flatten()

    state_variables = sym_plant_components.state_variables
    for i in tqdm(range(num_timesteps), desc="Computing W from W_sym"):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_to_val[state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_to_val[state_variables.q_ddot[j]] = joint_data.joint_accelerations[i, j]
        W_data[i * num_joints : (i + 1) * num_joints, :] = Evaluate(W_sym, sym_to_val)

    print(f"Condition number: {np.linalg.cond(W_data)}")

    alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, sym_parameters_arr, tau_data, sym_plant_components


def extract_data_matrix_symbolic_Wensing_trick(
    prog: Optional[MathematicalProgram] = None,
    use_one_link_arm: bool = False,
):
    """
    Wensing's trick for computing W_sym by setting lumped parameters equal to one at a
    time. This doesn't work as Drake doesn't simplify the expressions and thus throws a
    division by zero error for terms such as m * hx/m when setting m = 0.
    Simplifying using sympy should be possible but faces the same slowness issues as
    `extract_data_matrix_symbolic`.
    """
    urdf_path = (
        "./models/one_link_arm.urdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )
    num_joints = 1 if use_one_link_arm else 7
    time_step = 1e-3

    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=time_step
    )
    sym_plant_components = create_symbolic_plant(
        arm_components=arm_components, prog=prog, use_lumped_parameters=True
    )

    sym_parameters_arr = np.concatenate(
        [params.get_lumped_param_list() for params in sym_plant_components.parameters]
    )

    forces = MultibodyForces_[Expression](sym_plant_components.plant)
    sym_plant_components.plant.CalcForceElementsContribution(
        sym_plant_components.plant_context, forces
    )
    sym_torques = sym_plant_components.plant.CalcInverseDynamics(
        sym_plant_components.plant_context,
        sym_plant_components.state_variables.q_ddot.T,
        forces,
    )

    W_column_vectors = []
    for i in range(len(sym_parameters_arr)):
        param_values = np.zeros(len(sym_parameters_arr))
        param_values[i] = 1.0
        W_column_vector = []
        expression: Expression
        for expression in sym_torques:
            W_column_vector.append(
                expression.EvaluatePartial(dict(zip(sym_parameters_arr, param_values)))
            )
        W_column_vectors.append(W_column_vector)
    W_sym = np.hstack(W_column_vectors)

    print("W_sym:\n", W_sym)

    # Substitute data values and compute least squares fit
    joint_data = get_data(num_q=num_joints, plant=arm_components.plant)
    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * 10
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    tau_data = joint_data.joint_torques.flatten()

    state_variables = sym_plant_components.state_variables
    for i in tqdm(range(num_timesteps), desc="Computing W from W_sym"):
        sym_to_val = {}
        for j in range(num_joints):
            sym_to_val[state_variables.q[j]] = joint_data.joint_positions[i, j]
            sym_to_val[state_variables.q_dot[j]] = joint_data.joint_velocities[i, j]
            sym_to_val[state_variables.q_ddot[j]] = joint_data.joint_accelerations[i, j]
        W_data[i * num_joints : (i + 1) * num_joints, :] = Evaluate(W_sym, sym_to_val)

    print(f"Condition number: {np.linalg.cond(W_data)}")

    alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, sym_parameters_arr, tau_data, sym_plant_components


if __name__ == "__main__":
    extract_data_matrix_symbolic(use_one_link_arm=False)
    # extract_data_matrix_symbolic_Wensing_trick(use_one_link_arm=True)
