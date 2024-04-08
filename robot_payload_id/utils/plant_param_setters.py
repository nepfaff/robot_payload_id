import logging

from typing import Dict, Optional

import numpy as np

from pydrake.all import (
    Context,
    JointActuator,
    MultibodyPlant,
    RevoluteJoint,
    RigidBody,
    SpatialInertia,
    UnitInertia,
)

from robot_payload_id.utils import ArmComponents, ArmPlantComponents


def write_parameters_to_plant(
    arm_components: ArmComponents,
    var_name_param_dict: Dict[str, float],
    plant_context: Optional[Context] = None,
) -> ArmPlantComponents:
    """
    Create a plant context that has the parameters written to it. The plant is assumed
    to contain a robot arm.
    """
    plant = arm_components.plant
    plant_context = (
        plant.CreateDefaultContext() if plant_context is None else plant_context
    )
    # Check if only contains payload parameters
    payload_only = len(var_name_param_dict) == 10

    for i in range(arm_components.num_joints):
        if payload_only and i < arm_components.num_joints - 1:
            # Skip all but the last link
            continue

        try:
            link: RigidBody = plant.GetBodyByName(f"iiwa_link_{i+1}")
            joint: RevoluteJoint = plant.GetJointByName(f"iiwa_joint_{i+1}")
            joint_actuator: JointActuator = plant.GetJointActuatorByName(
                f"iiwa_joint_{i+1}"
            )
        except:
            link: RigidBody = plant.GetBodyByName(f"link{i + 1}")
            joint: RevoluteJoint = plant.GetJointByName(f"joint{i+1}")
            joint_actuator: JointActuator = plant.GetJointActuatorByName(f"joint{i+1}")

        mass = abs(var_name_param_dict[f"m{i}(0)"])
        # TODO: Project the inertia to the closest valid one
        link.SetSpatialInertiaInBodyFrame(
            plant_context,
            SpatialInertia(
                mass=mass,
                p_PScm_E=np.array(
                    [
                        var_name_param_dict[f"hx{i}(0)"],
                        var_name_param_dict[f"hy{i}(0)"],
                        var_name_param_dict[f"hz{i}(0)"],
                    ]
                )
                / mass,
                G_SP_E=UnitInertia(
                    Ixx=var_name_param_dict[f"Ixx{i}(0)"] / mass,
                    Iyy=var_name_param_dict[f"Iyy{i}(0)"] / mass,
                    Izz=var_name_param_dict[f"Izz{i}(0)"] / mass,
                    Ixy=var_name_param_dict[f"Ixy{i}(0)"] / mass,
                    Ixz=var_name_param_dict[f"Ixz{i}(0)"] / mass,
                    Iyz=var_name_param_dict[f"Iyz{i}(0)"] / mass,
                ),
                skip_validity_check=True,
            ),
        )
        set_rotor_inertia = False
        if f"rotor_inertia{i}(0)" in var_name_param_dict:
            joint_actuator.SetRotorInertia(
                plant_context, abs(var_name_param_dict[f"rotor_inertia{i}(0)"])
            )
            set_rotor_inertia = True
        if f"reflected_inertia{i}(0)" in var_name_param_dict:
            if set_rotor_inertia:
                logging.warning(
                    "Both rotor inertia and reflected inertia are set. Ignoring rotor "
                    + "inertia."
                )
            gear_ratio = joint_actuator.default_gear_ratio()
            reflected_inertia = var_name_param_dict[f"reflected_inertia{i}(0)"]
            rotor_inertia = reflected_inertia / gear_ratio**2
            joint_actuator.SetRotorInertia(plant_context, rotor_inertia)
        if f"viscous_friction{i}(0)" in var_name_param_dict:
            joint.SetDamping(
                plant_context,
                abs(var_name_param_dict[f"viscous_friction{i}(0)"]),
            )
        if f"dynamic_dry_friction{i}(0)" in var_name_param_dict:
            logging.warning(
                "Dynamic dry friction not implemented yet in Drake. Skipping."
            )

    return ArmPlantComponents(plant=plant, plant_context=plant_context)


def write_parameters_to_rigid_body(
    plant: MultibodyPlant,
    var_name_param_dict: Dict[str, float],
    body_name: str,
    plant_context: Optional[Context] = None,
) -> ArmComponents:
    """
    Create a plant context that has the parameters written to it. The parameters are
    of a single rigid body in the plant.

    `var_name_param_dict` is a dictionary that contains the 10 inertial parameters.
    """
    # Create a plant context
    plant_context = (
        plant.CreateDefaultContext() if plant_context is None else plant_context
    )
    body: RigidBody = plant.GetBodyByName(body_name)

    # Extract the parameters
    for name, value in var_name_param_dict.items():
        if "m" in name:
            mass = abs(value)
        if "hx" in name:
            hx = value
        if "hy" in name:
            hy = value
        if "hz" in name:
            hz = value
        if "Ixx" in name:
            Ixx = value
        if "Iyy" in name:
            Iyy = value
        if "Izz" in name:
            Izz = value
        if "Ixy" in name:
            Ixy = value
        if "Ixz" in name:
            Ixz = value
        if "Iyz" in name:
            Iyz = value

    # TODO: Project the inertia to the closest valid one
    body.SetSpatialInertiaInBodyFrame(
        plant_context,
        SpatialInertia(
            mass=mass,
            p_PScm_E=np.array([hx, hy, hz]) / mass,
            G_SP_E=UnitInertia(
                Ixx=Ixx / mass,
                Iyy=Iyy / mass,
                Izz=Izz / mass,
                Ixy=Ixy / mass,
                Ixz=Ixz / mass,
                Iyz=Iyz / mass,
            ),
            skip_validity_check=True,
        ),
    )

    return ArmPlantComponents(plant=plant, plant_context=plant_context)
