from typing import List

import pydrake.symbolic as sym

from pydrake.all import SpatialInertia_, UnitInertia_

from robot_payload_id.utils import (
    ArmComponents,
    JointParameters,
    SymbolicArmPlantComponents,
    SymJointStateVariables,
)


def create_symbolic_plant(
    arm_components: ArmComponents,
) -> SymbolicArmPlantComponents:
    """Creates a symbolic plant for a robotic arm system.

    Args:
        arm_components (ArmComponents): The components of the robotic arm system.

    Returns:
        SymbolicArmPlantComponents: The symbolic plant and associated symbolic
        components.
    """
    sym_plant = arm_components.plant.ToSymbolic()
    sym_plant_context = sym_plant.CreateDefaultContext()

    # Create the state variables
    q = sym.MakeVectorVariable(arm_components.num_joints, "q")
    q_dot = sym.MakeVectorVariable(arm_components.num_joints, "\dot{q}")
    q_ddot = sym.MakeVectorVariable(arm_components.num_joints, "\ddot{q}")
    tau = sym.MakeVectorVariable(arm_components.num_joints, "\tau")
    sym_state_variables = SymJointStateVariables(
        q=q, q_dot=q_dot, q_ddot=q_ddot, tau=tau
    )

    sym_plant.get_actuation_input_port().FixValue(sym_plant_context, tau)
    sym_plant.SetPositions(sym_plant_context, q)
    sym_plant.SetVelocities(sym_plant_context, q_dot)

    # Create the parameters
    sym_parameters: List[JointParameters] = []
    for i in range(arm_components.num_joints):
        m = sym.Variable(f"m_{i + 1}")
        cx = sym.Variable(f"c_{{x_{i + 1}}}")
        cz = sym.Variable(f"c_{{z_{i + 1}}}")
        Gyy = sym.Variable(f"G_{{yy_{i + 1}}}")
        sym_parameters.append(JointParameters(m=m, cx=cx, cz=cz, Gyy=Gyy))

        link = sym_plant.GetBodyByName(f"link{i + 1}")

        link.SetMass(sym_plant_context, m)

        # Set CoM and moment of inertia
        current_spatial_inertia = link.CalcSpatialInertiaInBodyFrame(sym_plant_context)
        com = current_spatial_inertia.get_com()
        com[0] = cx
        com[2] = cz
        unit_inertia = current_spatial_inertia.get_unit_inertia().get_moments()
        unit_inertia[1] = Gyy
        inertia = SpatialInertia_[sym.Expression](
            m,
            com,
            UnitInertia_[sym.Expression](
                unit_inertia[0], unit_inertia[1], unit_inertia[2]
            ),
            skip_validity_check=False,
        )
        link.SetSpatialInertiaInBodyFrame(sym_plant_context, inertia)

    return SymbolicArmPlantComponents(
        plant=sym_plant,
        plant_context=sym_plant_context,
        state_variables=sym_state_variables,
        parameters=sym_parameters,
    )
