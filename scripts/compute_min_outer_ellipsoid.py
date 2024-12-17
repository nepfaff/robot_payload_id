from typing import Tuple

import numpy as np
import trimesh

from pydrake.all import AffineBall


def extract_quadratic_form(affine_ball: AffineBall) -> np.ndarray:
    """
    Extract the Q matrix representing the ellipsoid in quadratic form.
    See https://drake.mit.edu/pydrake/pydrake.geometry.optimization.html#pydrake.geometry.optimization.AffineBall.

    :param affine_ball: AffineBall object from Drake
    :return: Q (4x4 numpy array)
    """
    B = affine_ball.B()  # Affine transformation matrix (3x3)
    center = affine_ball.center().flatten()  # Center of the ellipsoid (3,)

    # Compute A as the inverse of B
    A = np.linalg.inv(B)
    AT_A = A.T @ A  # A^T A

    # Build the Q matrix
    Q = np.zeros((4, 4))
    Q[:3, :3] = AT_A
    Q[:3, 3] = -AT_A @ center
    Q[3, :3] = -center.T @ AT_A
    Q[3, 3] = center.T @ AT_A @ center - 1

    return Q


def drake_ellipsoid(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the minimum enclosing ellipsoid using Drake's optimization.
    :param vertices: np.ndarray of shape (n, 3) - mesh vertices.
    :return: center (np.ndarray), radii (np.ndarray), axes (np.ndarray), Q (np.ndarray)
    """
    points = vertices.T  # Transpose to shape (3, n)
    affine_ball = AffineBall.MinimumVolumeCircumscribedEllipsoid(points)

    center = affine_ball.center().flatten()
    B = affine_ball.B()
    radii = np.linalg.norm(B, axis=0)
    axes = B / radii

    # Extract Q matrix
    Q = extract_quadratic_form(affine_ball)

    return center, radii, axes, Q


def compute_min_ellipsoid(mesh_file: str) -> None:
    """
    Load a triangle mesh and compute the Drake-based enclosing ellipsoid.
    :param mesh_file: Path to the mesh file.
    """
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_file)
    vertices = mesh.vertices

    # Compute Drake-based enclosing ellipsoid
    center, radii, axes, Q = drake_ellipsoid(vertices)

    print("Ellipsoid Center:", center)
    print("Ellipsoid Radii:", radii)
    print("Ellipsoid Axes:\n", axes)
    print("Ellipsoid Quadratic Form (Q):\n", Q)

    # Visualization using trimesh
    ellipsoid = trimesh.creation.icosphere(subdivisions=3)
    ellipsoid.apply_scale(radii)

    # Build transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = axes
    transform[:3, 3] = center
    ellipsoid.apply_transform(transform)

    # Set ellipsoid transparency and color
    ellipsoid.visual.face_colors = [0, 0, 255, 100]  # Blue with transparency
    mesh.visual.face_colors = [200, 200, 200, 255]  # Light gray opaque

    scene = trimesh.Scene([mesh, ellipsoid])
    scene.show()


if __name__ == "__main__":
    compute_min_ellipsoid("models/test_object.obj")
