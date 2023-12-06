from functools import partial
from typing import List, Optional

import pydrake.symbolic as sym

from pydrake.all import MathematicalProgram, RigidBody, SpatialInertia_, UnitInertia_

from robot_payload_id.utils import (
    ArmComponents,
    JointParameters,
    SymbolicArmPlantComponents,
    SymJointStateVariables,
)


def create_symbolic_plant(
    arm_components: ArmComponents,
    prog: Optional[MathematicalProgram] = None,
) -> SymbolicArmPlantComponents:
    """Creates a symbolic plant for a robotic arm system.

    Args:
        arm_components (ArmComponents): The components of the robotic arm system.
        prog (MathematicalProgram): An optional MathematicalProgram to use for variable
            creation.

    Returns:
        SymbolicArmPlantComponents: The symbolic plant and associated symbolic
        components.
    """
    sym_plant = arm_components.plant.ToSymbolic()
    sym_plant_context = sym_plant.CreateDefaultContext()

    # Create the state variables
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

    # Create the parameters
    make_variable_func = (
        partial(prog.NewContinuousVariables, 1)
        if prog is not None
        else partial(sym.MakeVectorVariable, 1)
    )
    sym_parameters: List[JointParameters] = []
    for i in range(arm_components.num_joints):
        m = make_variable_func(f"m_{i}")[0]
        cx = make_variable_func(f"c_{{x_{i}}}")[0]
        cy = make_variable_func(f"c_{{y_{i}}}")[0]
        cz = make_variable_func(f"c_{{z_{i}}}")[0]
        Gxx = make_variable_func(f"G_{{xx_{i}}}")[0]
        Gxy = make_variable_func(f"G_{{xy_{i}}}")[0]
        Gxz = make_variable_func(f"G_{{xz_{i}}}")[0]
        Gyy = make_variable_func(f"G_{{yy_{i}}}")[0]
        Gyz = make_variable_func(f"G_{{yz_{i}}}")[0]
        Gzz = make_variable_func(f"G_{{zz_{i}}}")[0]

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
            )
        )

        link: RigidBody = sym_plant.GetBodyByName(f"iiwa_link_{i}")
        # link: RigidBody = sym_plant.GetBodyByName(f"link{i + 1}")

        link.SetMass(sym_plant_context, m)

        # Set CoM and moment of inertia
        com = [cx, cy, cz]
        unit_inertia = UnitInertia_[sym.Expression](
            Ixx=Gxx, Ixy=Gxy, Ixz=Gxz, Iyy=Gyy, Iyz=Gyz, Izz=Gzz
        )
        inertia = SpatialInertia_[sym.Expression](
            m,
            com,
            unit_inertia,
            skip_validity_check=False,
        )
        link.SetSpatialInertiaInBodyFrame(sym_plant_context, inertia)

    return SymbolicArmPlantComponents(
        plant=sym_plant,
        plant_context=sym_plant_context,
        state_variables=sym_state_variables,
        parameters=sym_parameters,
    )
