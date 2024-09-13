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


def get_candidate_sys_id_welded_subgraphs(plant) -> List[List[RigidBody]]:
    subgraphs = get_welded_subgraphs(plant)
    candidate_subgraphs: List[List[RigidBody]] = []
    for subgraph in subgraphs:
        first_body = subgraph[0]
        # Ignore world subgraph
        if are_frames_welded(plant, first_body.body_frame(), plant.world_frame()):
            continue

        # Ignore subgraphs with no mass
        masses = [body.default_mass() for body in subgraph]
        if sum(masses) == 0.0:
            continue

        # Return subgraph
        candidate_subgraphs.append(subgraph)
    return candidate_subgraphs


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
    """
    Returns mass, center of mass, and rotational inertia of a body in the body frame:
    M_BBo_B spatial inertia of this rigid body B about Bo (B's origin), expressed in B.
    """
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
    payload_only: bool,
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
        payload_only (bool): Whether to include only the 10 inertial parameters of the
            last link.

    Returns:
        List[JointParameters]: A list of inertial parameters for each joint in the
            plant.
    """
    assert not (
        add_rotor_inertia and add_reflected_inertia
    ), "Cannot add rotor inertia and reflected inertia at the same time."

    # bodies = get_candidate_sys_id_bodies(plant)
    subgraphs = get_candidate_sys_id_welded_subgraphs(plant)
    joints = get_revolute_joints(plant)
    joint_actuators = get_joint_actuators(plant)

    if payload_only:
        subgraphs = subgraphs[-1:]
        joints = joints[-1:]
        joint_actuators = joint_actuators[-1:]

    joint_params: List[JointParameters] = []
    for subgraph, joint, joint_actuator in zip(subgraphs, joints, joint_actuators):
        # Combine inertial parameters of all bodies in subgraph
        if len(subgraph) == 1:
            # Parameters in the body frame.
            combined_mass, combined_com, combined_rot_inertia = get_body_inertial_param(
                subgraph[0], context
            )
        else:
            # Express parameters in the frame of the first body.

            # Calculate the spatial inertia of the subgraph about B1o (origin of the
            # first body in the subgraph), expressed in B1 (frame of the first body in
            # the subgraph).
            B1_frame = subgraph[0].body_frame()
            M_SB1o_B1 = plant.CalcSpatialInertia(
                context=context,
                frame_F=B1_frame,
                body_indexes=[body.index() for body in subgraph],
            )
            combined_mass = M_SB1o_B1.get_mass()
            combined_com = M_SB1o_B1.get_com()
            combined_rot_inertia = M_SB1o_B1.CalcRotationalInertia().CopyToFullMatrix3()

        joint_params.append(
            JointParameters(
                m=combined_mass,
                cx=combined_com[0],
                cy=combined_com[1],
                cz=combined_com[2],
                hx=combined_mass * combined_com[0],
                hy=combined_mass * combined_com[1],
                hz=combined_mass * combined_com[2],
                Ixx=combined_rot_inertia[0, 0],
                Iyy=combined_rot_inertia[1, 1],
                Izz=combined_rot_inertia[2, 2],
                Ixy=combined_rot_inertia[0, 1],
                Ixz=combined_rot_inertia[0, 2],
                Iyz=combined_rot_inertia[1, 2],
                rotor_inertia=joint_actuator.rotor_inertia(context)
                if add_rotor_inertia and not payload_only
                else None,
                reflected_inertia=joint_actuator.rotor_inertia(context)
                * joint_actuator.gear_ratio(context) ** 2
                if add_reflected_inertia and not payload_only
                else None,
                viscous_friction=joint.GetDamping(context)
                if add_viscous_friction and not payload_only
                else None,
                dynamic_dry_friction=0.0
                if add_dynamic_dry_friction and not payload_only
                else None,  # Not currently modelled by Drake
            )
        )

    return joint_params
