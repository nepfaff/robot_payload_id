from typing import List, Optional

import numpy as np

from drake_torch_sys_id_test import add_model_iiwa
from pydrake.all import (
    AutoDiffXd,
    DecomposeLumpedParameters,
    Evaluate,
    Expression,
    ForceElementIndex,
    MathematicalProgram,
    MultibodyForces_,
    MultibodyPlant,
    Solve,
)
from tqdm import tqdm

from robot_payload_id.environment import create_arm
from robot_payload_id.symbolic import create_autodiff_plant, create_symbolic_plant
from robot_payload_id.utils import (
    ArmComponents,
    ArmPlantComponents,
    JointData,
    JointParameters,
)


def symbolic_decomposition_with_dynamic_substitution(
    sym_plant_components: ArmPlantComponents, joint_data: JointData
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
            [params.get_base_param_list() for params in sym_plant_components.parameters]
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
    alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
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
        alpha_fit = np.linalg.lstsq(W_data, -w0_data, rcond=None)[0]
    else:
        alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, alpha_sym, tau_data, sym_plant_components


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
        q += 0.01 * np.random.normal(size=dim_q)
        v += 0.01 * np.random.normal(size=dim_q)
        vd += 0.01 * np.random.normal(size=dim_q)
        tau_gt += 0.01 * np.random.normal(size=dim_q)

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


def extract_data_matrix_autodiff(use_one_link_arm: bool = False):
    num_joints = 1 if use_one_link_arm else 7
    time_step = 0
    urdf_path = (
        "./models/one_link_arm.urdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )
    arm_components = create_arm(
        arm_file_path=urdf_path, num_joints=num_joints, time_step=time_step
    )
    joint_data = get_data(
        num_q=num_joints, plant=arm_components.plant, N=1000, add_noise=False
    )

    ad_plant_components = create_autodiff_plant(arm_components=arm_components)

    ad_parameters_arr = np.concatenate(
        [params.get_lumped_param_list() for params in ad_plant_components.parameters]
    )

    num_timesteps = len(joint_data.sample_times_s)
    num_lumped_params = num_joints * 10
    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    # w0_data = np.zeros(num_timesteps * num_joints)
    tau_data = joint_data.joint_torques.flatten()

    for i in tqdm(range(num_timesteps)):
        # Set joint data
        ad_plant_components.plant.get_actuation_input_port().FixValue(
            ad_plant_components.plant_context, joint_data.joint_torques[i]
        )
        ad_plant_components.plant.SetPositions(
            ad_plant_components.plant_context, joint_data.joint_positions[i]
        )
        ad_plant_components.plant.SetVelocities(
            ad_plant_components.plant_context, joint_data.joint_velocities[i]
        )

        # Implicit dynamics
        # derivatives = (
        #     ad_plant_components.plant_context.Clone().get_mutable_continuous_state()
        # )
        # print("derivatives", derivatives.get_mutable_vector())
        # print("size", derivatives.size())
        # derivatives.SetFromVector(
        #     np.hstack(
        #         (
        #             0 * joint_data.joint_positions[i],
        #             joint_data.joint_accelerations[i],
        #         )
        #     )
        # )
        # residual = ad_plant_components.plant.CalcImplicitTimeDerivativesResidual(
        #     ad_plant_components.plant_context, derivatives
        # )
        # implicit_dynamics = residual[int(len(residual) / 2) :]

        # Inverse dynamics
        forces = MultibodyForces_[AutoDiffXd](ad_plant_components.plant)
        ad_plant_components.plant.CalcForceElementsContribution(
            ad_plant_components.plant_context, forces
        )
        sym_torques = ad_plant_components.plant.CalcInverseDynamics(
            ad_plant_components.plant_context,
            joint_data.joint_accelerations[i],
            forces,
        )

        # Differentiate w.r.t. parameters
        sym_torques_derivative = np.vstack(
            [joint_torques.derivatives() for joint_torques in sym_torques]
        )
        W_data[i * num_joints : (i + 1) * num_joints, :] = sym_torques_derivative

    print(f"Condition number: {np.linalg.cond(W_data)}")
    alpha_fit = np.linalg.lstsq(W_data, tau_data, rcond=None)[0]
    print(f"alpha_fit: {alpha_fit}")

    return W_data, tau_data, ad_plant_components


def main_symbolic():
    prog = MathematicalProgram()

    # NOTE: All methods achieve the same results for the one link arm but the pure
    # symbolic methods does not scale to the iiwa arm
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
    # Group lumped parameters to ensure convexity
    lumped_vars = prog.NewContinuousVariables(len(alpha_sym), "alpha")
    for lumped_var, old_var in zip(lumped_vars, alpha_sym, strict=True):
        # TODO: Want to define this to be true instead of adding a constraint (constraint not linear!)
        prog.AddLinearEqualityConstraint(lumped_var == old_var)

    # NOTE: This fails as vars appear in combinations (lumped parameters)
    prog.Add2NormSquaredCost(A=W_data, b=tau_data, vars=lumped_vars)
    for pseudo_inertia in pseudo_inertias:
        # NOTE: This fails due to nonlinearities (lumped parameters)
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-3 * np.identity(4))


def main_ad():
    use_one_link_arm = False
    num_links = 1 if use_one_link_arm else 7
    W_data, tau_data, _ = extract_data_matrix_autodiff(
        use_one_link_arm=use_one_link_arm
    )

    prog = MathematicalProgram()

    # Create decision variables
    variables: List[JointParameters] = []
    for i in range(num_links):
        m = prog.NewContinuousVariables(1, f"m{i}")[0]
        hx = prog.NewContinuousVariables(1, f"hx{i}")[0]
        hy = prog.NewContinuousVariables(1, f"hy{i}")[0]
        hz = prog.NewContinuousVariables(1, f"hz{i}")[0]
        Ixx = prog.NewContinuousVariables(1, f"Ixx{i}")[0]
        Ixy = prog.NewContinuousVariables(1, f"Ixy{i}")[0]
        Ixz = prog.NewContinuousVariables(1, f"Ixz{i}")[0]
        Iyy = prog.NewContinuousVariables(1, f"Iyy{i}")[0]
        Iyz = prog.NewContinuousVariables(1, f"Iyz{i}")[0]
        Izz = prog.NewContinuousVariables(1, f"Izz{i}")[0]

        variables.append(
            JointParameters(
                m=m,
                hx=hx,
                hy=hy,
                hz=hz,
                Ixx=Ixx,
                Ixy=Ixy,
                Ixz=Ixz,
                Iyy=Iyy,
                Iyz=Iyz,
                Izz=Izz,
            )
        )

    # Construct pseudo inertia
    pseudo_inertias = []
    for i in range(num_links):
        mass = np.array([[variables[i].m]])
        inertia = variables[i].get_inertia_matrix()
        density_weighted_2nd_moment_matrix = (
            0.5 * np.trace(inertia) * np.eye(3) - inertia
        )
        density_weighted_1st_moment = np.array(
            [variables[i].hx, variables[i].hy, variables[i].hz]
        ).reshape((3, 1))
        pseudo_inertias.append(
            np.block(
                [
                    [
                        density_weighted_2nd_moment_matrix,
                        density_weighted_1st_moment,
                    ],
                    [
                        density_weighted_1st_moment.T,
                        mass,
                    ],
                ]
            )
        )

    # Solve constrained SDP
    variable_list = np.concatenate([var.get_lumped_param_list() for var in variables])
    # TODO: Define norm with respect to error covariance matrix from unconstrained fit
    # (This will probably require AddQuadraticCost)
    prog.Add2NormSquaredCost(A=W_data, b=tau_data, vars=variable_list)
    # TODO: Replace this with geometric regularization
    regularization_weight = 1e-3
    prog.AddQuadraticCost(
        regularization_weight * variable_list.T @ variable_list, is_convex=True
    )
    for pseudo_inertia in pseudo_inertias:
        prog.AddPositiveSemidefiniteConstraint(pseudo_inertia - 1e-6 * np.identity(4))

    from pydrake.all import CommonSolverOption, SolverOptions

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)

    result = Solve(prog, None, options)
    if result.is_success():
        print("SDP result:\n", result.GetSolution(variable_list))
    else:
        print("Failed to solve")
        print(result.get_solution_result())
        print(
            result.get_solver_details().rescode,
            result.get_solver_details().solution_status,
        )

    with open("iiwa_id_sdp_latex.txt", "w") as f:
        f.write(prog.ToLatex())

    print(np.linalg.eigvalsh(W_data.T @ W_data))


if __name__ == "__main__":
    # extract_data_matrix_autodiff(use_one_link_arm=False)
    # extract_data_matrix_symbolic(use_one_link_arm=True)
    main_ad()
