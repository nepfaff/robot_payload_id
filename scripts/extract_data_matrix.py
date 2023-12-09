from typing import Optional

import numpy as np

from drake_torch_sys_id_test import add_model_iiwa
from pydrake.all import (
    DecomposeLumpedParameters,
    Evaluate,
    Expression,
    MathematicalProgram,
    MultibodyForces_,
    MultibodyPlant,
)
from tqdm import tqdm

from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_symbolic_plant
from robot_payload_id.utils import ArmComponents, JointData, SymbolicArmPlantComponents


def symbolic_decomposition_with_dynamic_substitution(
    sym_plant_components: SymbolicArmPlantComponents, joint_data: JointData
):
    """Algorithm 1 from Andy's thesis."""
    num_timesteps = len(joint_data.sample_times_s)
    num_links = sym_plant_components.state_variables.q.shape[-1]
    num_lumped_params = (
        51  # num_links * 10  # TODO: True in theory but maybe not in practice
    )
    W_data = np.zeros((num_timesteps * num_links, num_lumped_params))
    # tau_data = np.zeros((num_timesteps * num_links, 1))
    tau_data = joint_data.joint_torques.flatten()

    plant = sym_plant_components.plant
    state_variables = sym_plant_components.state_variables
    for i in tqdm(range(num_timesteps)):
        # Substitute symbolic variables with numeric values
        new_context = sym_plant_components.plant_context.Clone()
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
            [params.get_param_list() for params in sym_plant_components.parameters]
        )
        W, alpha, w0 = DecomposeLumpedParameters(sym_torques, sym_parameters_arr)

        try:
            W_data[i * num_links : (i + 1) * num_links, :] = Evaluate(W, {})
            alpha_sym = alpha
        except:
            print("W_data shape:", W_data.shape)
            print("W shape:", W.shape)
            print("i:", i)
            print("------------------")
            continue
        # tau_data[i * num_links : (i + 1) * num_links] = Evaluate(sym_torques, {})

    print(f"Condition number: {np.linalg.cond(W_data)}")
    alpha_fit = np.linalg.lstsq(W_data, tau_data)[0]
    print(f"alpha: {alpha}")
    print(f"alpha_fit: {alpha_fit}")

    return W_data, alpha_sym, tau_data


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
        arm_components=arm_components, prog=prog
    )

    sym_parameters_arr = np.concatenate(
        [params.get_param_list() for params in sym_plant_components.parameters]
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

    print("W:\n", W_sym)
    print("alpha:\n", alpha_sym)
    print("w0:\n", w0_sym)

    # Substitute data values and compute least squares fit
    joint_data = get_data(num_q=num_joints, plant=arm_components.plant)
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

    print(f"Condition number: {np.linalg.cond(W_data)}")

    if use_w0:
        alpha_fit = np.linalg.lstsq(W_data, -w0_data)[0]
    else:
        alpha_fit = np.linalg.lstsq(W_data, tau_data)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, alpha_sym, tau_data, sym_plant_components


def get_data(plant, num_q) -> JointData:
    N = 100
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

    joint_data = JointData(
        joint_positions=q,
        joint_velocities=v,
        joint_accelerations=vd,
        joint_torques=tau_gt,
        sample_times_s=np.arange(N) * 1e-3,
    )
    return joint_data


def extract_data_matrix_andy_iteration(prog: Optional[MathematicalProgram] = None):
    num_q = 7
    plant = MultibodyPlant(time_step=1e-3)
    bodies = add_model_iiwa(plant)
    # bodies = add_model_one_link_arm(plant)
    plant.Finalize()
    joint_data = get_data(num_q=num_q, plant=plant)
    arm_components = ArmComponents(
        num_joints=num_q,
        diagram=None,
        plant=plant,
        trajectory_source=None,
        state_logger=None,
        commanded_torque_logger=None,
        meshcat=None,
        meshcat_visualizer=None,
    )
    sym_plant_components = create_symbolic_plant(
        arm_components=arm_components, prog=prog
    )

    W_data, alpha_sym, tau_data = symbolic_decomposition_with_dynamic_substitution(
        sym_plant_components=sym_plant_components, joint_data=joint_data
    )
    return W_data, alpha_sym, tau_data, sym_plant_components


def main():
    prog = MathematicalProgram()

    # NOTE: Both methods achieve the same results for the one link arm but only the
    # iteration one scales to the iiwa
    # W_data, alpha_sym, tau_data, sym_plant_components = extract_data_matrix_symbolic(
    #     prog=prog
    # )
    (
        W_data,
        alpha_sym,
        tau_data,
        sym_plant_components,
    ) = extract_data_matrix_andy_iteration(prog=prog)

    # Construct pseudo inertia
    pseudo_inertias = []
    params = sym_plant_components.parameters
    for i in range(7):
        inertia = params[i].get_inertia_matrix()
        density_weighted_2nd_moment_matrix = (
            0.5 * np.trace(inertia) * np.eye(3) - inertia
        )
        density_weighted_1st_moment = params[i].m * np.array(
            [params[i].cx, params[i].cy, params[i].cz]
        )
        pseudo_inertias.append(
            np.block(
                [
                    [
                        density_weighted_2nd_moment_matrix,
                        density_weighted_1st_moment.reshape((3, 1)),
                    ],
                    [
                        density_weighted_1st_moment.reshape((3, 1)).T,
                        np.array([[params[i].m]]),
                    ],
                ]
            )
        )

    # Solve constrained SDP
    # NOTE: This fails as vars appear in combinations (lumped parameters)
    prog.Add2NormSquaredCost(A=W_data, b=tau_data, vars=alpha_sym)
    for pseudo_inertia in pseudo_inertias:
        # NOTE: This fails due to nonlinearities (lumped parameters)
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-3 * np.identity(4))


if __name__ == "__main__":
    main()
