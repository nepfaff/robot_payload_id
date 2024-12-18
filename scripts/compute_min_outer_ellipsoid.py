import numpy as np
import trimesh

from robot_payload_id.utils.utils import drake_ellipsoid


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
