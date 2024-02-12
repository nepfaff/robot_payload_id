from functools import partial
from typing import List, Optional

import numpy as np
import pydrake.symbolic as sym

from pydrake.all import (
    AutoDiffXd,
    ExtractValue,
    JointActuator,
    MathematicalProgram,
    MultibodyPlant,
    RevoluteJoint,
    RigidBody,
    RotationalInertia_,
    SpatialInertia,
    SpatialInertia_,
    UnitInertia_,
)

from robot_payload_id.utils import (
    ArmComponents,
    ArmPlantComponents,
    JointParameters,
    SymJointStateVariables,
)


def create_symbolic_plant(
    arm_components: ArmComponents,
    prog: Optional[MathematicalProgram] = None,
    use_lumped_parameters: bool = False,
) -> ArmPlantComponents:
    """Creates a symbolic plant for a robotic arm system.

    Args:
        arm_components (ArmComponents): The components of the robotic arm system.
        prog (MathematicalProgram): An optional MathematicalProgram to use for variable
            creation.
        use_lumped_parameters (bool): Whether to create symbolic parameters for the
            lumped parameters h = m * c and I = m * G that the inverse dynamics are
            linear in. If False, creates symbolic parameters for m, c, and G.

    Returns:
        ArmPlantComponents: The symbolic plant and associated symbolic components.
    """
    # Create the symbolic plant
    sym_plant: MultibodyPlant = arm_components.plant.ToSymbolic()
    sym_plant_context = sym_plant.CreateDefaultContext()

    # Create the symbolic state variables
    make_vec_variable_func = (
        prog.NewContinuousVariables if prog is not None else sym.MakeVectorVariable
    )
    q = make_vec_variable_func(arm_components.num_joints, "q")
    q_dot = make_vec_variable_func(arm_components.num_joints, "\dot{q}")
    q_ddot = make_vec_variable_func(arm_components.num_joints, "\ddot{q}")
    tau = make_vec_variable_func(arm_components.num_joints, "\tau")
    sym_state_variables = SymJointStateVariables(
        q=q, q_dot=q_dot, q_ddot=q_ddot, tau=tau
    )

    sym_plant.get_actuation_input_port().FixValue(sym_plant_context, tau)
    sym_plant.SetPositions(sym_plant_context, q)
    sym_plant.SetVelocities(sym_plant_context, q_dot)

    # Create the symbolic inertial parameters
    # NOTE: Using sym.Variable instead of partial(sym.MakeVectorVariable, 1) would lead
    # to less notation clutter but is harder to combine with MathematicalProgram variables
    make_variable_func = (
        partial(prog.NewContinuousVariables, 1)
        if prog is not None
        else partial(sym.MakeVectorVariable, 1)
    )
    sym_parameters: List[JointParameters] = []
    for i in range(arm_components.num_joints):
        m = make_variable_func(f"m_{i}")[0]
        if use_lumped_parameters:
            hx = make_variable_func(f"h_{{x_{i}}}")[0]
            hy = make_variable_func(f"h_{{y_{i}}}")[0]
            hz = make_variable_func(f"h_{{z_{i}}}")[0]
            Ixx = make_variable_func(f"I_{{xx_{i}}}")[0]
            Ixy = make_variable_func(f"I_{{xy_{i}}}")[0]
            Ixz = make_variable_func(f"I_{{xz_{i}}}")[0]
            Iyy = make_variable_func(f"I_{{yy_{i}}}")[0]
            Iyz = make_variable_func(f"I_{{yz_{i}}}")[0]
            Izz = make_variable_func(f"I_{{zz_{i}}}")[0]

            cx = hx / m
            cy = hy / m
            cz = hz / m
            Gxx = Ixx / m
            Gxy = Ixy / m
            Gxz = Ixz / m
            Gyy = Iyy / m
            Gyz = Iyz / m
            Gzz = Izz / m
        else:
            cx = make_variable_func(f"c_{{x_{i}}}")[0]
            cy = make_variable_func(f"c_{{y_{i}}}")[0]
            cz = make_variable_func(f"c_{{z_{i}}}")[0]
            Gxx = make_variable_func(f"G_{{xx_{i}}}")[0]
            Gxy = make_variable_func(f"G_{{xy_{i}}}")[0]
            Gxz = make_variable_func(f"G_{{xz_{i}}}")[0]
            Gyy = make_variable_func(f"G_{{yy_{i}}}")[0]
            Gyz = make_variable_func(f"G_{{yz_{i}}}")[0]
            Gzz = make_variable_func(f"G_{{zz_{i}}}")[0]

            hx = m * cx
            hy = m * cy
            hz = m * cz
            Ixx = m * Gxx
            Ixy = m * Gxy
            Ixz = m * Gxz
            Iyy = m * Gyy
            Iyz = m * Gyz
            Izz = m * Gzz

        sym_parameters.append(
            JointParameters(
                m=m,
                cx=cx,
                cy=cy,
                cz=cz,
                Gxx=Gxx,
                Gxy=Gxy,
                Gxz=Gxz,
                Gyy=Gyy,
                Gyz=Gyz,
                Gzz=Gzz,
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

        try:
            # There is no hope to identify link 0, so we skip it
            link: RigidBody = sym_plant.GetBodyByName(f"iiwa_link_{i+1}")
        except:
            link: RigidBody = sym_plant.GetBodyByName(f"link{i + 1}")

        # m_val = link.get_mass(sym_plant_context)
        # cx_val, cy_val, cz_val = link.CalcCenterOfMassInBodyFrame(sym_plant_context)
        # spatial_inertia: SpatialInertia = link.CalcSpatialInertiaInBodyFrame(
        #     sym_plant_context
        # )
        # print(m_val, cx_val, cy_val, cz_val, "\n",spatial_inertia.get_unit_inertia().CopyToFullMatrix3(),"\n",spatial_inertia.CalcRotationalInertia().CopyToFullMatrix3())
        # print("-----------------")

        # Add the symbolic parameters to the plant
        # com = [cx, cy, cz]
        # unit_inertia = UnitInertia_[sym.Expression](
        #     Ixx=Gxx, Ixy=Gxy, Ixz=Gxz, Iyy=Gyy, Iyz=Gyz, Izz=Gzz
        # )
        # inertia = SpatialInertia_[sym.Expression](
        #     m, com, unit_inertia, skip_validity_check=False
        # )
        inertia: SpatialInertia = SpatialInertia_[
            sym.Expression
        ]().MakeFromLumpedParameters(
            m,
            [hx, hy, hz],
            RotationalInertia_[sym.Expression](Ixx, Iyy, Izz, Ixy, Ixz, Iyz),
        )
        # print("Calling SetSpatialInertiaInBodyFrame with inertia:\n", inertia)
        link.SetSpatialInertiaInBodyFrame(sym_plant_context, inertia)

    return ArmPlantComponents(
        plant=sym_plant,
        plant_context=sym_plant_context,
        state_variables=sym_state_variables,
        parameters=sym_parameters,
    )


def create_autodiff_plant(
    arm_components: ArmComponents,
    add_rotor_inertia: bool,
    add_viscous_friction: bool = False,
    add_dynamic_dry_friction: bool = False,
) -> ArmPlantComponents:
    """Creates an autodiff plant for a robotic arm system.

    Args:
        arm_components (ArmComponents): The components of the robotic arm system.
        add_rotor_inertia (bool): Whether to add autodiff reflected rotor inertia
            parameters.
        add_viscous_friction (bool): Whether to add autodiff viscous friction
            parameters. NOTE: Drake does not yet support viscous friction as part of
            the context. Hence, this parameter is created but not added to the plant.
        add_dynamic_dry_friction (bool): Whether to add autodiff dynamic dry friction
            parameters. NOTE: Drake does not yet support dynamic dry friction as part of
            the context. Hence, this parameter is created but not added to the plant.

    Returns:
        ArmPlantComponents: The autodiff plant and associated autodiff components.
    """
    # Create an autodiff plant
    ad_plant: MultibodyPlant = arm_components.plant.ToAutoDiffXd()
    ad_plant_context = ad_plant.CreateDefaultContext()

    # Create the autodiff parameters
    ad_parameters: List[JointParameters] = []
    num_params_per_joint = (
        10 + add_rotor_inertia + add_viscous_friction + add_dynamic_dry_friction
    )
    num_params = arm_components.num_joints * num_params_per_joint
    for i in range(arm_components.num_joints):
        try:
            # There is no hope to identify link 0, so we skip it
            link: RigidBody = ad_plant.GetBodyByName(f"iiwa_link_{i+1}")
            joint: RevoluteJoint = ad_plant.GetJointByName(f"iiwa_joint_{i+1}")
            joint_actuator: JointActuator = ad_plant.GetJointActuatorByName(
                f"iiwa_joint_{i+1}"
            )
        except:
            link: RigidBody = ad_plant.GetBodyByName(f"link{i + 1}")
            joint: RevoluteJoint = ad_plant.GetJointByName(f"joint{i+1}")
            joint_actuator: JointActuator = ad_plant.GetJointActuatorByName(
                f"joint{i+1}"
            )

        m_val = link.get_mass(ad_plant_context).value()

        # Extract the current values from the plant
        cx_val, cy_val, cz_val = ExtractValue(
            link.CalcCenterOfMassInBodyFrame(ad_plant_context)
        )
        spatial_inertia: SpatialInertia = link.CalcSpatialInertiaInBodyFrame(
            ad_plant_context
        )
        Ixx_val = ExtractValue(spatial_inertia.CopyToFullMatrix6()[:3, :3])[0, 0]
        Ixy_val = ExtractValue(spatial_inertia.CopyToFullMatrix6()[:3, :3])[0, 1]
        Ixz_val = ExtractValue(spatial_inertia.CopyToFullMatrix6()[:3, :3])[0, 2]
        Iyy_val = ExtractValue(spatial_inertia.CopyToFullMatrix6()[:3, :3])[1, 1]
        Iyz_val = ExtractValue(spatial_inertia.CopyToFullMatrix6()[:3, :3])[1, 2]
        Izz_val = ExtractValue(spatial_inertia.CopyToFullMatrix6()[:3, :3])[2, 2]

        # Create autodiff variables for the inertial parameters
        m_vec = np.zeros(num_params)
        m_vec[(i * num_params_per_joint)] = 1
        m_ad = AutoDiffXd(m_val, m_vec)

        cx_vec = np.zeros(num_params)
        cx_vec[(i * num_params_per_joint) + 1] = 1
        hx_ad = AutoDiffXd(cx_val * m_val, cx_vec)
        cx_ad = hx_ad / m_ad
        cy_vec = np.zeros(num_params)
        cy_vec[(i * num_params_per_joint) + 2] = 1
        hy_ad = AutoDiffXd(cy_val * m_val, cy_vec)
        cy_ad = hy_ad / m_ad
        cz_vec = np.zeros(num_params)
        cz_vec[(i * num_params_per_joint) + 3] = 1
        hz_ad = AutoDiffXd(cz_val * m_val, cz_vec)
        cz_ad = hz_ad / m_ad
        com_ad = [cx_ad, cy_ad, cz_ad]

        Ixx_vec = np.zeros(num_params)
        Ixx_vec[(i * num_params_per_joint) + 4] = 1
        Ixx_ad = AutoDiffXd(Ixx_val, Ixx_vec)
        Gxx_ad = Ixx_ad / m_ad
        Ixy_vec = np.zeros(num_params)
        Ixy_vec[(i * num_params_per_joint) + 5] = 1
        Ixy_ad = AutoDiffXd(Ixy_val, Ixy_vec)
        Gxy_ad = Ixy_ad / m_ad
        Ixz_vec = np.zeros(num_params)
        Ixz_vec[(i * num_params_per_joint) + 6] = 1
        Ixz_ad = AutoDiffXd(Ixz_val, Ixz_vec)
        Gxz_ad = Ixz_ad / m_ad
        Iyy_vec = np.zeros(num_params)
        Iyy_vec[(i * num_params_per_joint) + 7] = 1
        Iyy_ad = AutoDiffXd(Iyy_val, Iyy_vec)
        Gyy_ad = Iyy_ad / m_ad
        Iyz_vec = np.zeros(num_params)
        Iyz_vec[(i * num_params_per_joint) + 8] = 1
        Iyz_ad = AutoDiffXd(Iyz_val, Iyz_vec)
        Gyz_ad = Iyz_ad / m_ad
        Izz_vec = np.zeros(num_params)
        Izz_vec[(i * num_params_per_joint) + 9] = 1
        Izz_ad = AutoDiffXd(Izz_val, Izz_vec)
        Gzz_ad = Izz_ad / m_ad
        G_ad = UnitInertia_[AutoDiffXd](Gxx_ad, Gyy_ad, Gzz_ad, Gxy_ad, Gxz_ad, Gyz_ad)

        offset = 10
        if add_rotor_inertia:
            rotor_inertia_vec = np.zeros(num_params)
            rotor_inertia_vec[(i * num_params_per_joint) + offset] = 1
            rotor_inertia_ad = AutoDiffXd(
                joint_actuator.default_rotor_inertia(), rotor_inertia_vec
            )
            offset += 1
        if add_viscous_friction:
            viscous_friction_vec = np.zeros(num_params)
            viscous_friction_vec[(i * num_params_per_joint) + offset] = 1
            viscous_friction_ad = AutoDiffXd(joint.damping(), viscous_friction_vec)
            offset += 1
        if add_dynamic_dry_friction:
            dynamic_dry_friction_vec = np.zeros(num_params)
            dynamic_dry_friction_vec[(i * num_params_per_joint) + offset] = 1
            dynamic_dry_friction_ad = AutoDiffXd(0.0, dynamic_dry_friction_vec)

        ad_parameters.append(
            JointParameters(
                m=m_ad,
                cx=cx_ad,
                cy=cy_ad,
                cz=cz_ad,
                Gxx=Gxx_ad,
                Gxy=Gxy_ad,
                Gxz=Gxz_ad,
                Gyy=Gyy_ad,
                Gyz=Gyz_ad,
                Gzz=Gzz_ad,
                hx=hx_ad,
                hy=hy_ad,
                hz=hz_ad,
                Ixx=Ixx_ad,
                Ixy=Ixy_ad,
                Ixz=Ixz_ad,
                Iyy=Iyy_ad,
                Iyz=Iyz_ad,
                Izz=Izz_ad,
                rotor_inertia=rotor_inertia_ad if add_rotor_inertia else None,
                viscous_friction=viscous_friction_ad if add_viscous_friction else None,
                dynamic_dry_friction=dynamic_dry_friction_ad
                if add_dynamic_dry_friction
                else None,
            )
        )

        # Add the autodiff parameters to the plant
        spatial_inertia_ad = SpatialInertia_[AutoDiffXd](
            m_ad, com_ad, G_ad, skip_validity_check=True
        )
        link.SetSpatialInertiaInBodyFrame(ad_plant_context, spatial_inertia_ad)
        if add_rotor_inertia:
            joint_actuator.SetRotorInertia(ad_plant_context, rotor_inertia_ad)
        # if add_viscous_friction:
        # Not yet implemented in Drake (see
        # https://github.com/RobotLocomotion/drake/issues/14405)
        # joint.SetDamping(ad_plant_context, viscous_friction_ad)

    return ArmPlantComponents(
        plant=ad_plant,
        plant_context=ad_plant_context,
        state_variables=None,
        parameters=ad_parameters,
    )
