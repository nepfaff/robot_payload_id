import os

from manipulation.utils import ConfigureParser
from pydrake.all import MultibodyPlant, Parser


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("models/package.xml"))
    return parser
