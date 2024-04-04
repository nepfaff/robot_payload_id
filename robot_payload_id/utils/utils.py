import os

from typing import Any, List

from manipulation.utils import ConfigureParser
from pydrake.all import MathematicalProgram, MultibodyPlant, Parser


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("models/package.xml"))
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
