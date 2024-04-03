## Similar to `InertiaVisualizer` in drake but allows editing the plant's inertial
## parameters.

import argparse
import logging

from pathlib import Path

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BodyIndex,
    DiagramBuilder,
    Ellipsoid,
    IllustrationProperties,
    MeshcatVisualizer,
    MultibodyPlant,
    Rgba,
    RoleAssign,
    Simulator,
    SpatialInertia,
    StartMeshcat,
)

from robot_payload_id.utils import (
    ArmComponents,
    ArmPlantComponents,
    get_parser,
    write_parameters_to_plant,
    write_parameters_to_rigid_body,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_param_path",
        type=Path,
        help="Path to the initial parameter `.npy` file. If not provided, the initial "
        + "parameters are set to the model's parameters.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("./models/iiwa.dmd.yaml"),
        help="Path to the model file to visualize the ellipsoids for.",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=7,
        help="Number of joints in the model. If zero, then the model is assumed to be "
        + "a rigid body.",
    )
    parser.add_argument(
        "--existing_model_alpha",
        type=float,
        default=0.5,
        help="Alpha value for the existing model's visual geometries.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )

    args = parser.parse_args()
    initial_param_path = args.initial_param_path
    model_path: Path = args.model_path
    num_joints = args.num_joints
    existing_model_alpha = args.existing_model_alpha

    logging.basicConfig(level=args.log_level)
    np.random.seed(0)

    # Create plant
    builder = DiagramBuilder()
    plant: MultibodyPlant
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = get_parser(plant)
    models = parser.AddModels(model_path.as_posix())

    # Disable gravity to prevent robot from falling down
    for model in models:
        plant.set_gravity_enabled(model, False)

    plant.Finalize()

    # Make existing visual geometries transparent
    for i in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(i))
        visual_geometry_ids = plant.GetVisualGeometriesForBody(body)
        for geometry_id in visual_geometry_ids:
            old_props = scene_graph.model_inspector().GetIllustrationProperties(
                geometry_id
            )
            new_props = IllustrationProperties(old_props)
            old_rgba = old_props.GetProperty("phong", "diffuse")
            new_props.UpdateProperty(
                group_name="phong",
                name="diffuse",
                value=Rgba(
                    r=old_rgba.r(),
                    g=old_rgba.g(),
                    b=old_rgba.b(),
                    a=existing_model_alpha,
                ),
            )
            scene_graph.AssignRole(
                source_id=plant.get_source_id(),
                geometry_id=geometry_id,
                properties=new_props,
                assign=RoleAssign.kReplace,
            )

    # Create meshcat
    meshcat = StartMeshcat()
    _ = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Load parameters
    if initial_param_path is not None:
        logging.info(f"Loading initial parameters from {initial_param_path}.")
        var_sol_dict = np.load(initial_param_path, allow_pickle=True).item()

        if num_joints == 0:
            arm_plant_components = write_parameters_to_rigid_body(
                plant, var_sol_dict, plant_context
            )
        else:
            arm_components = ArmComponents(
                num_joints=num_joints,
                diagram=None,
                plant=plant,
                trajectory_source=None,
                state_logger=None,
                commanded_torque_logger=None,
                meshcat=None,
                meshcat_visualizer=None,
            )
            arm_plant_components = write_parameters_to_plant(
                arm_components, var_sol_dict, plant_context
            )
    else:
        logging.info(f"Using default parameters from {model_path}.")
        arm_plant_components = ArmPlantComponents(
            plant=plant,
            plant_context=plant_context,
        )

    # Create and visualize ellipsoids
    for i in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(i))
        M_BBo_B: SpatialInertia = body.CalcSpatialInertiaInBodyFrame(
            arm_plant_components.plant_context
        )
        mass = M_BBo_B.get_mass()
        if np.isnan(mass):
            continue

        (
            radii,
            X_BE,
        ) = M_BBo_B.CalcPrincipalSemiDiametersAndPoseForSolidEllipsoid()

        # Clip for improved visualization. See CalculateInertiaGeometry in drake
        # internal.
        new_radii = np.clip(radii, a_min=1e-2 * radii.max(), a_max=None)
        if np.any(new_radii != radii):
            logging.warning(
                f"Radii for body {body.name()} were clipped to {new_radii}."
            )

        # Assuming the ~density of water for the visualization ellipsoid, scale up
        # the ellipsoid representation of the unit inertia to have a volume that
        # matches the body's actual mass, so that our ellipsoid actually has the
        # same inertia as M_BBo_B. (We're illustrating M_BBo, not G_BBo.)
        density = 1000.0
        unit_inertia_ellipsoid_mass = density * 4.0 / 3.0 * np.pi * np.prod(new_radii)
        volume_scale = mass / unit_inertia_ellipsoid_mass
        abc = new_radii * np.cbrt(volume_scale)
        ellipsoid = Ellipsoid(abc)

        # Compute transform
        X_WB = plant.EvalBodyPoseInWorld(plant_context, body)
        X_WE = X_WB.multiply(X_BE)

        # Visualize
        color = np.random.choice(range(255), size=3) / 255
        meshcat.SetObject(
            path=f"ellipsoid_{i}",
            shape=ellipsoid,
            rgba=Rgba(r=color[0], g=color[1], b=color[2], a=1.0),
        )
        meshcat.SetTransform(path=f"ellipsoid_{i}", X_ParentPath=X_WE)

    # Simulate
    simulator = Simulator(diagram, context=context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(5.0)


if __name__ == "__main__":
    main()
