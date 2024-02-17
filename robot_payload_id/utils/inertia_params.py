import logging

from typing import Callable, List, Optional, Tuple

import numpy as np

from pydrake.all import (
    BodyIndex,
    Context,
    Frame,
    JointActuator,
    JointActuatorIndex,
    JointIndex,
    ModelInstanceIndex,
    MultibodyPlant,
    RevoluteJoint,
    RigidBody,
    SpatialInertia,
)

from .dataclasses import JointParameters


def _get_plant_aggregate(
    num_func: Callable,
    get_func: Callable,
    index_cls: Callable,
    model_instances: Optional[List[ModelInstanceIndex]] = None,
) -> list:
    items = []
    for i in range(num_func()):
        item = get_func(index_cls(i))
        if model_instances is None or item.model_instance() in model_instances:
            items.append(item)
    return items


def get_revolute_joints(
    plant: MultibodyPlant, model_instances: Optional[List[ModelInstanceIndex]] = None
) -> List[RevoluteJoint]:
    joints = _get_plant_aggregate(
        plant.num_joints, plant.get_joint, JointIndex, model_instances
    )
    return [joint for joint in joints if isinstance(joint, RevoluteJoint)]


def get_joint_actuators(
    plant: MultibodyPlant, model_instances: Optional[List[ModelInstanceIndex]] = None
) -> List[JointActuator]:
    return _get_plant_aggregate(
        plant.num_actuators,
        plant.get_joint_actuator,
        JointActuatorIndex,
        model_instances,
    )


def get_bodies(
    plant: MultibodyPlant, model_instances: Optional[List[ModelInstanceIndex]] = None
) -> List[RigidBody]:
    return _get_plant_aggregate(
        plant.num_bodies, plant.get_body, BodyIndex, model_instances
    )


def get_welded_subgraphs(plant: MultibodyPlant) -> List[List[RigidBody]]:
    bodies = get_bodies(plant)
    bodies_seen = set()
    subgraphs: List[List[RigidBody]] = []
    for body in bodies:
        subgraph: List[RigidBody] = plant.GetBodiesWeldedTo(body)
        if body not in bodies_seen:
            subgraphs.append(subgraph)
        bodies_seen |= set(subgraph)
    return subgraphs


def are_frames_welded(plant: MultibodyPlant, A: Frame, B: Frame) -> bool:
    if A.body() is B.body():
        return True
    A_bodies = plant.GetBodiesWeldedTo(A.body())
    return B.body() in A_bodies


def get_candidate_sys_id_bodies(plant) -> List[RigidBody]:
    subgraphs = get_welded_subgraphs(plant)
    bodies: List[RigidBody] = []
    for subgraph in subgraphs:
        first_body = subgraph[0]
        # Ignore world subgraph
        if are_frames_welded(plant, first_body.body_frame(), plant.world_frame()):
            continue

        # Ignore subgraphs with no mass
        masses = [body.default_mass() for body in subgraph]
        if sum(masses) == 0.0:
            continue

        # Choose single body to represent subgraph
        if len(subgraph) == 1:
            (body,) = subgraph
        else:
            # Take body with greatest mass and optimize it.
            # TODO: Should we instead distribute more evenly?
            logging.warning(
                "Multiple bodies in subgraph. Taking heaviest. The GT inertial "
                + "parameters and model computed GT torques will be wrong!"
            )
            i = np.argmax(masses)
            body = subgraph[i]
        bodies.append(body)
    return bodies


def extract_inertial_param(
    spatial_inertia: SpatialInertia,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Extracts mass, center of mass, and rotational inertia from spatial inertia."""
    mass = spatial_inertia.get_mass()
    com = spatial_inertia.get_com()
    rot_inertia = spatial_inertia.CalcRotationalInertia()
    rot_inertia = rot_inertia.CopyToFullMatrix3()
    return mass, com, rot_inertia


def get_body_inertial_param(
    body: RigidBody, context: Context
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns mass, center of mass, and rotational inertia of a body."""
    spatial_inertia = body.CalcSpatialInertiaInBodyFrame(context)
    mass, com, rot_inertia = extract_inertial_param(spatial_inertia)
    return mass, com, rot_inertia


def get_plant_joint_params(
    plant: MultibodyPlant,
    context: Context,
    add_rotor_inertia: bool,
    add_reflected_inertia: bool,
    add_viscous_friction: bool,
    add_dynamic_dry_friction: bool,
) -> List[JointParameters]:
    """Returns the parameters for all joints and their associated welded subgraph in the
    plant.

    Args:
        plant (MultibodyPlant): The plant to extract inertial parameters from.
        context (Context): The context to extract inertial parameters from.
        add_rotor_inertia (bool): Whether to add rotor inertia to the joint parameters.
        add_reflected_inertia (bool): Whether to add reflected inertia to the joint
            parameters. NOTE: This is mutually exclusive with `add_rotor_inertia`.
        add_viscous_friction (bool): Whether to add viscous friction to the joint
            parameters.
        add_dynamic_dry_friction (bool): Whether to add dynamic dry friction to the
            joint parameters.

    Returns:
        List[JointParameters]: A list of inertial parameters for each joint in the
            plant.
    """
    assert not (
        add_rotor_inertia and add_reflected_inertia
    ), "Cannot add rotor inertia and reflected inertia at the same time."

    bodies = get_candidate_sys_id_bodies(plant)
    joints = get_revolute_joints(plant)
    joint_actuators = get_joint_actuators(plant)

    joint_params: List[JointParameters] = []
    for body, joint, joint_actuator in zip(bodies, joints, joint_actuators):
        mass, com, rot_inertia = get_body_inertial_param(body, context)
        joint_params.append(
            JointParameters(
                m=mass,
                cx=com[0],
                cy=com[1],
                cz=com[2],
                hx=mass * com[0],
                hy=mass * com[1],
                hz=mass * com[2],
                Ixx=rot_inertia[0, 0],
                Iyy=rot_inertia[1, 1],
                Izz=rot_inertia[2, 2],
                Ixy=rot_inertia[0, 1],
                Ixz=rot_inertia[0, 2],
                Iyz=rot_inertia[1, 2],
                rotor_inertia=joint_actuator.rotor_inertia(context)
                if add_rotor_inertia
                else None,
                reflected_inertia=joint_actuator.rotor_inertia(context)
                * joint_actuator.gear_ratio(context) ** 2
                if add_reflected_inertia
                else None,
                viscous_friction=joint.damping() if add_viscous_friction else None,
                dynamic_dry_friction=0.0
                if add_dynamic_dry_friction
                else None,  # Not currently modelled by Drake
            )
        )

    return joint_params
