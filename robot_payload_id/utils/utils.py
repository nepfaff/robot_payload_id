import os

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import trimesh

from manipulation.utils import ConfigureParser
from pydrake.all import AffineBall, MathematicalProgram, MultibodyPlant, Parser


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    ConfigureParser(parser)
    package_path = (
        Path(__file__).resolve().parent.parent.parent / "models" / "package.xml"
    )
    parser.package_map().AddPackageXml(filename=os.path.abspath(package_path))
    return parser


def get_package_xmls() -> List[str]:
    """Returns a list of package.xml files."""
    return [
        os.path.abspath("models/package.xml"),
    ]


def name_constraint(constraint_binding: "BindingTConstraintU", name: str) -> None:
    constraint = constraint_binding.evaluator()
    constraint.set_description(name)


def name_unnamed_constraints(prog: MathematicalProgram, name: str) -> None:
    """Assigns `name` to each unnamed constraint in `prog`."""
    constraint_bindings = prog.GetAllConstraints()
    constraints = [binding.evaluator() for binding in constraint_bindings]
    for constraint in constraints:
        if constraint.get_description() == "":
            constraint.set_description(name)


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def extract_quadratic_form(affine_ball: AffineBall) -> np.ndarray:
    """
    Extract the Q matrix representing the ellipsoid in quadratic form.
    See https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1optimization_1_1_affine_ball.html.
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


def compute_min_ellipsoid(
    mesh_file: str, transform: np.ndarray | None = None
) -> AffineBall:
    """
    Load a triangle mesh and compute the Drake-based enclosing ellipsoid.
    Args:
        mesh_file: Path to the mesh file.
        transform: Optional transformation matrix to apply to the mesh vertices.
    Returns:
        ellipsoid: AffineBall object representing the enclosing ellipsoid.
    """
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_file)
    assert isinstance(mesh, trimesh.Trimesh)
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    vertices = mesh.vertices
    # Compute Drake-based enclosing ellipsoid
    points = vertices.T  # Transpose to shape (3, n)
    ellipsoid = AffineBall.MinimumVolumeCircumscribedEllipsoid(points)
    return ellipsoid
