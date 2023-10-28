import os

from manipulation.utils import AddPackagePaths
from pydrake.all import MultibodyPlant, Parser


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().Add("robot_payload_id", os.path.abspath(""))
    return parser
