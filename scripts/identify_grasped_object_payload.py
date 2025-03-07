"""Identifies the robot parameters at multiple gripper openings.

The script identifies the inertial parameters of a grasped object and expresses them
in the object frame.

The script takes two data directories as input. The first directory contains the joint
data for the robot at multiple gripper openings. The second directory contains the
joint data for the grasped object.

Robot joint data directory structure:

<robot_joint_data_path>/gripper_position_<gripper_position>/identified_robot_params.npy

Object joint data directory structure:

<object_joint_data_path>/
    - joint_positions.npy of shape (T, 7)
    - joint_torques.npy of shape (T, 7)
    - sample_times_s.npy of shape (T,)
    - wsg_positions.npy of shape (T,)
    - manipuland_cloud_link7_frame.npy of shape (N, 3) is the manipuland pcd expressed
        in the link7 frame
T is the number of timesteps and N is the number of points in the point cloud.
"""

import argparse
import copy
import json
import logging

from pathlib import Path

import numpy as np
import open3d as o3d

from pydrake.all import (
    FixedOffsetFrame,
    MultibodyPlant,
    RigidTransform,
    SpatialInertia,
    UnitInertia,
)

from robot_payload_id.data import extract_numeric_data_matrix_autodiff
from robot_payload_id.optimization import solve_inertial_param_sdp
from robot_payload_id.utils import (
    ArmComponents,
    JointData,
    get_parser,
    get_plant_joint_params,
    process_joint_data,
    write_parameters_to_plant,
)
from robot_payload_id.utils.utils import compute_min_ellipsoid


def get_object_pose_in_link_frame(
    object_mesh_path: Path, object_pcd: np.ndarray, visualize: bool = False
) -> np.ndarray:
    """
    Get the pose of the object in the link frame.
    Uses RANSAC-based ICP to align the point cloud to the mesh to get the transform
    between them.

    Args:
        object_mesh_path: Path to the object mesh file that defines the object's
            frame.
        object_pcd: The point cloud of the object expressed in the link 7 frame of
            shape (N, 3).
        visualize: Whether to visualize the point cloud and mesh.

    Returns:
        The transform from the link frame to the object frame; i.e. X_LO.
    """
    # Load the target mesh and convert to point cloud
    target_mesh = o3d.io.read_triangle_mesh(str(object_mesh_path))
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=5000)

    # Convert numpy array to Open3D point cloud
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(object_pcd)

    # Voxel downsampling for both point clouds
    voxel_size = 0.005  # 5mm voxel size
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals for both point clouds
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )

    # Initial alignment using center of mass
    source_center = source_down.get_center()
    target_center = target_down.get_center()
    initial_translation = target_center - source_center

    init_transform = np.eye(4)
    init_transform[:3, 3] = initial_translation

    # Compute FPFH features
    def compute_fpfh_features(pcd, voxel_size):
        radius_normal = voxel_size * 2
        radius_feature = voxel_size * 5

        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    # Compute features for both point clouds
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)

    # Global registration using RANSAC with features
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=voxel_size * 5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                voxel_size * 5
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500),
    )

    # Try RANSAC result if successful, otherwise use initial translation
    if result_ransac.fitness > 0:
        initial_alignment = result_ransac.transformation
    else:
        initial_alignment = init_transform
        logging.warning("RANSAC failed, using center-of-mass alignment instead")

    # Refine with ICP using more lenient parameters
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance=voxel_size * 5,
        init=initial_alignment,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=500,
            relative_fitness=1e-7,
            relative_rmse=1e-7,
        ),
    )

    # Visualize the meshes before and after registration if requested
    if visualize:
        # Create frames for before registration
        source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        # Move source frame to source center
        source_frame_transform = np.eye(4)
        source_frame_transform[:3, 3] = source_center
        source_frame.transform(source_frame_transform)

        # Move target frame to target center
        target_frame_transform = np.eye(4)
        target_frame_transform[:3, 3] = target_center
        target_frame.transform(target_frame_transform)

        # Color the point clouds for visualization
        source_down.paint_uniform_color([1, 0, 0])  # Red for source
        target_down.paint_uniform_color([0, 1, 0])  # Green for target

        # Visualize before registration
        o3d.visualization.draw_geometries(
            [source_down, target_down, source_frame, target_frame],
            window_name="Before Registration",
        )

        # Create new frames for after registration
        source_frame_after = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05
        )
        source_frame_after.transform(source_frame_transform)  # Move to source center
        source_frame_after.transform(
            reg_p2p.transformation
        )  # Apply registration transform

        # Visualize after registration
        o3d.visualization.draw_geometries(
            [
                source_down.transform(reg_p2p.transformation),
                target_down,
                source_frame_after,
                target_frame,
            ],
            window_name="After Registration",
        )

        # Print debug information
        print(f"Source center: {source_center}")
        print(f"Target center: {target_center}")
        print(f"Initial translation: {initial_translation}")
        print(f"RANSAC transformation:\n{result_ransac.transformation}")
        print(f"Final ICP transformation:\n{reg_p2p.transformation}")
        print(f"RANSAC fitness: {result_ransac.fitness}")
        print(f"RANSAC inlier_rmse: {result_ransac.inlier_rmse}")
        print(f"ICP fitness: {reg_p2p.fitness}")
        print(f"ICP inlier_rmse: {reg_p2p.inlier_rmse}")

    # The transformation matrix from link frame to object frame
    X_LO = np.linalg.inv(reg_p2p.transformation)

    return X_LO


def visualize_object_inertia(
    object_mesh_path: Path,
    spatial_inertia: SpatialInertia,
    com: np.ndarray,
) -> None:
    """Visualizes the object mesh and its spatial inertia ellipsoid.

    Args:
        object_mesh_path: Path to the object mesh file.
        spatial_inertia: The spatial inertia of the object about its CoM, expressed in
            the object frame.
        com: The center of mass position in the object frame.
    """
    # Load the object mesh
    mesh = o3d.io.read_triangle_mesh(str(object_mesh_path))
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color

    # Create coordinate frames
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    com_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # Move COM frame to COM position
    com_frame_transform = np.eye(4)
    com_frame_transform[:3, 3] = com  # Use passed-in CoM
    com_frame.transform(com_frame_transform)

    # Calculate ellipsoid parameters
    radii, X_BE = spatial_inertia.CalcPrincipalSemiDiametersAndPoseForSolidEllipsoid()

    # Clip for improved visualization
    new_radii = np.clip(radii, a_min=1e-2 * radii.max(), a_max=None)
    if np.any(new_radii != radii):
        logging.warning(f"Radii were clipped to {new_radii}")

    # Scale ellipsoid to match mass
    density = 1000.0  # kg/m^3 (water density)
    unit_inertia_ellipsoid_mass = density * 4.0 / 3.0 * np.pi * np.prod(new_radii)
    volume_scale = spatial_inertia.get_mass() / unit_inertia_ellipsoid_mass
    abc = new_radii * np.cbrt(volume_scale)

    # Create sphere and scale it to ellipsoid
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    # Scale vertices to create ellipsoid
    vertices = np.asarray(sphere.vertices)
    vertices[:, 0] *= abc[0]
    vertices[:, 1] *= abc[1]
    vertices[:, 2] *= abc[2]

    # Apply rotation from X_BE
    R_BE = X_BE.rotation().matrix()
    vertices = vertices @ R_BE.T

    # Translate vertices to COM
    vertices += com

    # Update sphere vertices to create ellipsoid
    sphere.vertices = o3d.utility.Vector3dVector(vertices)
    sphere.paint_uniform_color([1, 0, 0])  # Red color

    # Visualize
    o3d.visualization.draw_geometries(
        [mesh, sphere, origin_frame, com_frame],
        window_name="Object Inertia Visualization",
        mesh_show_wireframe=True,
        mesh_show_back_face=True,
    )


def identify_grasped_object_payload(
    robot_joint_data_path: Path,
    object_joint_data_path: Path,
    object_mesh_path: Path,
    output_param_path: Path | None = None,
    vel_filter_order: int = 10,
    vel_cutoff_freq_hz: float = 5.6,
    acc_filter_order: int = 10,
    acc_cutoff_freq_hz: float = 4.2,
    torque_filter_order: int = 10,
    torque_cutoff_freq_hz: float = 4.0,
    visualize: bool = False,
    use_bounding_ellipsoid: bool = False,
    time_to_cutoff_at_beginning_s: float = 0.0,
    json_output_path: Path | None = None,
) -> None:
    """Identifies the inertial parameters of a grasped object and expresses them in the
    object frame.

    Args:
        robot_joint_data_path: Path to directory containing robot joint data at multiple
            gripper openings. See main file docstring for expected structure.
        object_joint_data_path: Path to directory containing object joint data. See
            main file docstring for expected structure.
        object_mesh_path: Path to the object mesh file. The identified params are
            expressed in this object's frame.
        output_param_path: Optional path to save the identified parameters as a .npy
            file.
        vel_filter_order: Order of the filter for joint velocities.
        vel_cutoff_freq_hz: Cutoff frequency for joint velocity filter.
        acc_filter_order: Order of the filter for joint accelerations.
        acc_cutoff_freq_hz: Cutoff frequency for joint acceleration filter.
        torque_filter_order: Order of the filter for joint torques.
        torque_cutoff_freq_hz: Cutoff frequency for joint torque filter.
        visualize: Whether to display debug visualizations.
        use_bounding_ellipsoid: Whether to use a bounding ellipsoid constraint for
            payload inertia.
        time_to_cutoff_at_beginning_s: Time to cutoff at the beginning of the data.
    """
    # Convert strings to paths if necessary.
    robot_joint_data_path = Path(robot_joint_data_path)
    object_joint_data_path = Path(object_joint_data_path)
    object_mesh_path = Path(object_mesh_path)

    # Get the payload/ object frame transform
    manipuland_cloud_link7_frame = np.load(
        object_joint_data_path / "manipuland_cloud_link7_frame.npy"
    )
    X_LO = get_object_pose_in_link_frame(
        object_mesh_path, manipuland_cloud_link7_frame, visualize=visualize
    )

    # Create arm and add the payload frame
    num_joints = 7
    # NOTE: This model must not have a payload attached. Otherwise, the w0 term will be
    # wrong and include the payload inertia.
    model_path = str(
        Path(__file__).resolve().parent.parent / "models" / "iiwa.dmd.yaml"
    )
    plant = MultibodyPlant(0.0)
    parser = get_parser(plant)
    parser.AddModels(model_path)
    last_link = plant.GetBodyByName("iiwa_link_7")
    payload_frame = FixedOffsetFrame(
        name="payload_frame",
        P=last_link.body_frame(),
        X_PF=RigidTransform(X_LO),
    )
    plant.AddFrame(payload_frame)
    plant.Finalize()

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

    # Determine closest gripper opening.
    available_gripper_positions = [
        float(folder.name.split("_")[-1])
        for folder in Path(robot_joint_data_path).iterdir()
        if folder.is_dir() and folder.name.startswith("gripper_position_")
    ]
    actual_gripper_positions = np.load(object_joint_data_path / "wsg_positions.npy")
    mean_gripper_position = np.mean(actual_gripper_positions)
    closest_gripper_position = min(
        available_gripper_positions, key=lambda x: abs(x - mean_gripper_position)
    )
    logging.info(
        f"Mean gripper position: {mean_gripper_position}. Using gripper position "
        f"{closest_gripper_position}."
    )

    # Load robot parameters.
    var_sol_dict = np.load(
        robot_joint_data_path
        / f"gripper_position_{closest_gripper_position}"
        / "identified_robot_params.npy",
        allow_pickle=True,
    ).item()
    arm_plant_components = write_parameters_to_plant(arm_components, var_sol_dict)

    # Load and process object joint data.
    raw_joint_data = JointData.load_from_disk_allow_missing(object_joint_data_path)
    raw_joint_data = JointData.cut_off_at_beginning(
        raw_joint_data, time_to_cutoff_at_beginning_s
    )
    joint_data = process_joint_data(
        joint_data=raw_joint_data,
        num_endpoints_to_remove=0,
        compute_velocities=True,
        filter_positions=False,
        pos_filter_order=0,
        pos_cutoff_freq_hz=0,
        vel_filter_order=vel_filter_order,
        vel_cutoff_freq_hz=vel_cutoff_freq_hz,
        acc_filter_order=acc_filter_order,
        acc_cutoff_freq_hz=acc_cutoff_freq_hz,
        torque_filter_order=torque_filter_order,
        torque_cutoff_freq_hz=torque_cutoff_freq_hz,
    )

    # Generate data matrix
    W_data, w0_data, _ = extract_numeric_data_matrix_autodiff(
        plant_components=arm_plant_components,
        joint_data=joint_data,
        add_rotor_inertia=False,
        add_reflected_inertia=True,
        add_viscous_friction=True,
        add_dynamic_dry_friction=True,
        payload_only=True,
    )
    tau_data = joint_data.joint_torques.flatten()
    # Transform from affine `tau = W * params + w0` into linear `(tau - w0) = W * params`
    tau_data -= w0_data

    # Construct the identified robot parameters without payload
    initial_last_link_params = get_plant_joint_params(
        arm_plant_components.plant,
        arm_plant_components.plant_context,
        add_rotor_inertia=False,
        add_reflected_inertia=True,
        add_viscous_friction=True,
        add_dynamic_dry_friction=True,
        payload_only=True,
    )[-1]

    if use_bounding_ellipsoid:
        # Compute the minimum bounding ellipsoid for the object
        bounding_ellipsoid = compute_min_ellipsoid(object_mesh_path, transform=X_LO)
    else:
        bounding_ellipsoid = None

    # Solve the SDP
    _, result, variable_names, variable_vec, _ = solve_inertial_param_sdp(
        num_links=num_joints,
        W_data=W_data,
        tau_data=tau_data,
        base_param_mapping=None,
        regularization_weight=0.0,
        params_guess=None,
        use_euclidean_regularization=False,
        identify_rotor_inertia=False,
        identify_reflected_inertia=True,
        identify_viscous_friction=True,
        identify_dynamic_dry_friction=True,
        payload_only=True,
        initial_last_link_params=initial_last_link_params,
        payload_bounding_ellipsoid=bounding_ellipsoid,
    )

    if result.is_success():
        final_cost = result.get_optimal_cost()
        logging.info(f"SDP cost: {final_cost}")
        var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_vec)))
        logging.info(f"SDP result:\n{var_sol_dict}")

        # Compute the difference in the last link's parameters. This corresponds
        # to the payload parameters if `initial_param_path` are the parameters
        # without payload.

        # We can subtract the lumped parameters as they are all in the last
        # link's frame.
        last_link_idx = num_joints - 1
        payload_mass = var_sol_dict[f"m{last_link_idx}(0)"] - initial_last_link_params.m
        payload_hcom = (
            np.array(
                [
                    var_sol_dict[f"hx{last_link_idx}(0)"],
                    var_sol_dict[f"hy{last_link_idx}(0)"],
                    var_sol_dict[f"hz{last_link_idx}(0)"],
                ]
            )
            - initial_last_link_params.m * initial_last_link_params.get_com()
        )
        payload_com = payload_hcom / payload_mass
        payload_rot_inertia = (
            np.array(
                [
                    [
                        var_sol_dict[f"Ixx{last_link_idx}(0)"],
                        var_sol_dict[f"Ixy{last_link_idx}(0)"],
                        var_sol_dict[f"Ixz{last_link_idx}(0)"],
                    ],
                    [
                        var_sol_dict[f"Ixy{last_link_idx}(0)"],
                        var_sol_dict[f"Iyy{last_link_idx}(0)"],
                        var_sol_dict[f"Iyz{last_link_idx}(0)"],
                    ],
                    [
                        var_sol_dict[f"Ixz{last_link_idx}(0)"],
                        var_sol_dict[f"Iyz{last_link_idx}(0)"],
                        var_sol_dict[f"Izz{last_link_idx}(0)"],
                    ],
                ]
            )
            - initial_last_link_params.get_inertia_matrix()
        )

        # Transform into the payload/ object frame
        plant_context = copy.deepcopy(arm_plant_components.plant_context)
        last_link.SetSpatialInertiaInBodyFrame(
            plant_context,
            SpatialInertia(
                mass=payload_mass,
                p_PScm_E=payload_com,
                G_SP_E=UnitInertia(
                    Ixx=payload_rot_inertia[0, 0] / payload_mass,
                    Iyy=payload_rot_inertia[1, 1] / payload_mass,
                    Izz=payload_rot_inertia[2, 2] / payload_mass,
                    Ixy=payload_rot_inertia[0, 1] / payload_mass,
                    Ixz=payload_rot_inertia[0, 2] / payload_mass,
                    Iyz=payload_rot_inertia[1, 2] / payload_mass,
                ),
            ),
        )
        # Spatial inertia of payload about the payload frame origin,
        # expressed in the payload frame.
        M_PPayload_Payload = arm_plant_components.plant.CalcSpatialInertia(
            context=plant_context,
            frame_F=plant.GetFrameByName("payload_frame"),
            body_indexes=[last_link.index()],
        )
        # Express inerita about CoM to match SDFormat convention.
        M_PPayloadcom_Payload = M_PPayload_Payload.Shift(M_PPayload_Payload.get_com())
        logging.info(
            "Difference in the last link's parameters (payload parameters). "
            + "Note that these are in identified object frame:"
        )
        payload_mass = M_PPayloadcom_Payload.get_mass()
        logging.info(f"Payload mass: {payload_mass}")
        com_PPayload_Payload = M_PPayload_Payload.get_com()
        logging.info(f"Payload CoM: {com_PPayload_Payload}")
        I_PPayloadcom_Payload = (
            M_PPayloadcom_Payload.CalcRotationalInertia().CopyToFullMatrix3()
        )
        logging.info("Payload inertia (about CoM):\n" + f"{I_PPayloadcom_Payload}\n")

        # Save inertial parameters to JSON.
        inertial_params = {
            "mass": float(payload_mass),
            "center_of_mass": com_PPayload_Payload.tolist(),
            "inertia_matrix": I_PPayloadcom_Payload.tolist(),
        }
        json_path = (
            json_output_path or object_joint_data_path.parent / "inertial_params.json"
        )
        with open(json_path, "w") as f:
            json.dump(inertial_params, f, indent=2)
        logging.info(f"Saved inertial parameters to {json_path}")

        if visualize:
            visualize_object_inertia(
                object_mesh_path=object_mesh_path,
                spatial_inertia=M_PPayloadcom_Payload,
                com=com_PPayload_Payload,  # Pass the original CoM
            )

        if output_param_path is not None:
            logging.info(f"Saving parameters to {output_param_path}")
            directory: Path = output_param_path.parent
            directory.mkdir(parents=True, exist_ok=True)
            np.save(output_param_path, var_sol_dict)
    else:
        logging.warning("Failed to solve inertial parameter SDP!")
        logging.info(f"Solution result:\n{result.get_solution_result()}")
        logging.info(f"Solver details:\n{result.get_solver_details()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_joint_data_path",
        required=True,
        type=Path,
        help="See main file docstring.",
    )
    parser.add_argument(
        "--object_joint_data_path",
        required=True,
        type=Path,
        help="See main file docstring.",
    )
    parser.add_argument(
        "--object_mesh_path",
        required=True,
        type=Path,
        help="Path to the object mesh file. The identified params are expressed in the "
        + "object frame.",
    )
    parser.add_argument(
        "--output_param_path",
        type=Path,
        default=None,
        help="Path to the output parameter `.npy` file. If not provided, the parameters "
        + "are not saved to disk.",
    )
    parser.add_argument(
        "--vel_order",
        type=int,
        default=10,
        help="The order of the filter for the joint velocities. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--vel_cutoff_freq_hz",
        type=float,
        default=5.6,
        help="The cutoff frequency of the filter for the joint velocities. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--acc_order",
        type=int,
        default=10,
        help="The order of the filter for the joint accelerations. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--acc_cutoff_freq_hz",
        type=float,
        default=4.2,
        help="The cutoff frequency of the filter for the joint accelerations. Only used "
        + "if `--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_order",
        type=int,
        default=10,
        help="The order of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--torque_cutoff_freq_hz",
        type=float,
        default=4.0,
        help="The cutoff frequency of the filter for the joint torques. Only used if "
        + "`--process_joint_data` is set.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to display debug visualizations.",
    )
    parser.add_argument(
        "--use_bounding_ellipsoid",
        action="store_true",
        help="Whether to use a bounding ellipsoid constraint for the payload inertia.",
    )
    parser.add_argument(
        "--time_to_cutoff_at_beginning_s",
        type=float,
        default=0.0,
        help="The time to cutoff at the beginning of the data.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    identify_grasped_object_payload(
        robot_joint_data_path=args.robot_joint_data_path,
        object_joint_data_path=args.object_joint_data_path,
        object_mesh_path=args.object_mesh_path,
        output_param_path=args.output_param_path,
        vel_filter_order=args.vel_order,
        vel_cutoff_freq_hz=args.vel_cutoff_freq_hz,
        acc_filter_order=args.acc_order,
        acc_cutoff_freq_hz=args.acc_cutoff_freq_hz,
        torque_filter_order=args.torque_order,
        torque_cutoff_freq_hz=args.torque_cutoff_freq_hz,
        visualize=args.visualize,
        use_bounding_ellipsoid=args.use_bounding_ellipsoid,
        time_to_cutoff_at_beginning_s=args.time_to_cutoff_at_beginning_s,
    )


if __name__ == "__main__":
    main()
