from dataclasses import dataclass
from typing import Dict

import numpy as np

from pydrake.all import (
    Context,
    RevoluteJoint,
    RigidBody,
    RotationalInertia,
    SpatialInertia,
    UnitInertia,
)


@dataclass
class JointDryFriction:
    max_generalized_force: float
    v0: float


@dataclass
class JointViscousFriction:
    damping: float


@dataclass
class JointFriction:
    dry: JointDryFriction
    viscous: JointViscousFriction


@dataclass
class JointParam:
    name: str
    reflected_rotor_inertia: float
    friction: JointFriction


@dataclass
class InertiaParam:
    Ixx: float
    Iyy: float
    Izz: float
    Ixy: float
    Ixz: float
    Iyz: float


@dataclass
class BodyParam:
    mass: float
    p_BBcm: np.ndarray
    I_BBo_B: InertiaParam


@dataclass
class ModelParam:
    joint_param: Dict[str, JointParam]
    body_param: Dict[str, BodyParam]


def ApplyJointParam(
    param: JointParam, context: Context, joint: RevoluteJoint, include_damping: bool
) -> None:
    if include_damping:
        joint.set_default_damping(param.friction.viscous.damping)
    else:
        joint.set_default_damping(0.0)


def ExtractJointParam(
    context: Context, joint: RevoluteJoint, include_damping: bool
) -> JointParam:
    return JointParam(
        name=joint.name(),
        reflected_rotor_inertia=0.0,
        friction=JointFriction(
            dry=JointDryFriction(max_generalized_force=0.0, v0=1.0),
            viscous=JointViscousFriction(
                damping=joint.damping() if include_damping else 0.0
            ),
        ),
    )


def RotationalInertiaToInertiaParam(inertia: RotationalInertia) -> InertiaParam:
    moments = inertia.get_moments()
    products = inertia.get_products()
    return InertiaParam(
        Ixx=moments(0),
        Iyy=moments(1),
        Izz=moments(2),
        Ixy=products(0),
        Ixz=products(1),
        Iyz=products(2),
    )


def BodyParamToSpatialInertia(param: BodyParam) -> SpatialInertia:
    I_BBo_B = RotationalInertiaToInertiaParam(param.I_BBo_B)
    G_BBo_B = UnitInertia(I_BBo_B / param.mass)
    M_BBo_b = SpatialInertia(mass=param.mass, p_PScm_E=param.p_BBcm, G_SP_E=G_BBo_B)
    return M_BBo_b


def SpatialInertiaToBodyParam(M_BBo_B: SpatialInertia) -> BodyParam:
    I_BBo_B = RotationalInertiaToInertiaParam(M_BBo_B.CalcRotationalInertia())
    return BodyParam(
        mass=M_BBo_B.get_mass(),
        p_BBcm=M_BBo_B.get_com(),
        I_BBo_B=I_BBo_B,
    )


def ApplyBodyParam(param: BodyParam, context: Context, body: RigidBody) -> None:
    M_BBo_B = BodyParamToSpatialInertia(param)
    body.SetSpatialInertiaInBodyFrame(context, M_BBo_B)


def ExtractBodyParam(context: Context, body: RigidBody) -> BodyParam:
    M_BBo_B = body.CalcSpatialInertiaInBodyFrame(context)
    return SpatialInertiaToBodyParam(M_BBo_B)
