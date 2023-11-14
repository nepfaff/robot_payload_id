import numpy as np

from pydrake.multibody.tree import BodyIndex, JointIndex


def _get_plant_aggregate(num_func, get_func, index_cls, model_instances=None):
    items = []
    for i in range(num_func()):
        item = get_func(index_cls(i))
        if model_instances is None or item.model_instance() in model_instances:
            items.append(item)
    return items


def get_bodies(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_bodies, plant.get_body, BodyIndex, model_instances
    )


def get_joints(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_joints, plant.get_joint, JointIndex, model_instances
    )


def are_frames_welded(plant, A, B):
    if A.body() is B.body():
        return True
    A_bodies = plant.GetBodiesWeldedTo(A.body())
    return B.body() in A_bodies
