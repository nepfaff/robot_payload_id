"""
Script for visualizing a model. The illustration, inertia, and proximity properties
are visualized but only the illustration is selected on startup.
"""

import argparse
import os

from pydrake.all import ModelVisualizer, StartMeshcat

from robot_payload_id.utils import get_package_xmls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model file to visualize the ellipsoids for.",
    )
    args = parser.parse_args()

    meshcat = StartMeshcat()
    visualizer = ModelVisualizer(meshcat=meshcat)
    for package_xml in get_package_xmls():
        paranet_dir = os.path.dirname(package_xml)
        visualizer.parser().package_map().PopulateFromFolder(paranet_dir)
    visualizer.parser().AddModels(args.model_path)
    visualizer.Run()


if __name__ == "__main__":
    main()
