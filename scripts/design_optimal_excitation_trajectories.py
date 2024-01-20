import argparse
import logging

from pathlib import Path

import numpy as np

from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import (
    CostFunction,
    ExcitationTrajectoryOptimizerBlackBoxNumeric,
    ExcitationTrajectoryOptimizerBlackBoxSymbolic,
    ExcitationTrajectoryOptimizerBlackBoxSymbolicNumeric,
    ExcitationTrajectoryOptimizerSnopt,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help="Use one link arm instead of 7-DOF arm.",
    )
    parser.add_argument(
        "--load_data_matrix",
        action="store_true",
        help="Load data matrix from file instead of computing it.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=["snopt", "black_box", "both"],
        help="Optimizer to use. If both, then use black-box to initialize SNOPT.",
    )
    parser.add_argument(
        "--cost_function",
        type=CostFunction,
        required=True,
        choices=list(CostFunction),
        help="Cost function to use.",
    )
    parser.add_argument(
        "--num_fourier_terms",
        type=int,
        required=True,
        help="Number of Fourier terms to use.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        required=True,
        help="Frequency of the trajectory.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        required=True,
        help="The number of timesteps to use.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1e-3,
        required=False,
        help="Trajectory timestep to use.",
    )
    parser.add_argument(
        "--snopt_iteration_limit",
        type=int,
        default=1000,
        required=False,
        help="Iteration limit for SNOPT.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
        required=False,
        help="Budget for black-box optimization.",
    )
    parser.add_argument(
        "--use_symbolic_computations",
        action="store_true",
        help="Whether to use symbolic computations. If False, then If False, then the "
        + "data matrix is numerically computed from scratch at each iteration.",
    )
    parser.add_argument(
        "--not_symbolically_reexpress_data_matrix",
        action="store_true",
        help="Whether to not symbolically re-express the data matrix. If True, then the "
        + "data matrix is numerically re-expressed at each iteration. Only used for "
        + "black-box optimization.",
    )
    parser.add_argument(
        "--logging_path",
        type=Path,
        default=None,
        required=False,
        help="Path to the directory to save the logs to. Only used for black-box "
        + "optimization.",
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

    use_one_link_arm = args.use_one_link_arm
    num_joints = 1 if use_one_link_arm else 7

    data_matrix_dir_path, model_path = None, None
    if args.load_data_matrix:
        data_matrix_dir_path = Path(
            "data/symbolic_data_matrix_one_link_arm"
            if use_one_link_arm
            else "data/symbolic_data_matrix_iiwa"
        )
    model_path = (
        "./models/one_link_arm.sdf" if use_one_link_arm else "./models/iiwa.dmd.yaml"
    )

    arm_components = create_arm(arm_file_path=model_path, num_joints=num_joints)
    plant = arm_components.plant
    robot_model_instance_idx = plant.GetModelInstanceByName("arm")

    optimizer = args.optimizer
    cost_function = args.cost_function
    num_fourier_terms = args.num_fourier_terms
    omega = args.omega
    num_timesteps = args.num_timesteps
    timestep = args.timestep
    snopt_iteration_limit = args.snopt_iteration_limit
    budget = args.budget
    logging_path = args.logging_path

    # Create the black-box optimizer
    if optimizer != "snopt":
        if (
            args.use_symbolic_computations
            and not args.not_symbolically_reexpress_data_matrix
        ):
            black_box_optimizer = ExcitationTrajectoryOptimizerBlackBoxSymbolic(
                num_joints=num_joints,
                cost_function=cost_function,
                num_fourier_terms=num_fourier_terms,
                omega=omega,
                num_timesteps=num_timesteps,
                timestep=timestep,
                plant=plant,
                robot_model_instance_idx=robot_model_instance_idx,
                budget=budget,
                logging_path=logging_path,
                data_matrix_dir_path=data_matrix_dir_path,
                model_path=model_path,
            )
        elif (
            args.use_symbolic_computations
            and args.not_symbolically_reexpress_data_matrix
        ):
            black_box_optimizer = ExcitationTrajectoryOptimizerBlackBoxSymbolicNumeric(
                num_joints=num_joints,
                cost_function=cost_function,
                num_fourier_terms=num_fourier_terms,
                omega=omega,
                num_timesteps=num_timesteps,
                timestep=timestep,
                plant=plant,
                robot_model_instance_idx=robot_model_instance_idx,
                budget=budget,
                logging_path=logging_path,
                data_matrix_dir_path=data_matrix_dir_path,
                model_path=model_path,
            )
        else:
            black_box_optimizer = ExcitationTrajectoryOptimizerBlackBoxNumeric(
                num_joints=num_joints,
                cost_function=cost_function,
                num_fourier_terms=num_fourier_terms,
                omega=omega,
                num_timesteps=num_timesteps,
                timestep=timestep,
                plant=plant,
                robot_model_instance_idx=robot_model_instance_idx,
                budget=budget,
                logging_path=logging_path,
                model_path=model_path,
            )
    if optimizer != "black_box":
        snopt_optimizer = ExcitationTrajectoryOptimizerSnopt(
            data_matrix_dir_path=data_matrix_dir_path,
            model_path=model_path,
            num_joints=num_joints,
            cost_function=cost_function,
            num_fourier_terms=num_fourier_terms,
            omega=omega,
            num_timesteps=num_timesteps,
            timestep=timestep,
            plant=plant,
            robot_model_instance_idx=robot_model_instance_idx,
            iteration_limit=snopt_iteration_limit,
        )

    if optimizer == "black_box":
        black_box_optimizer.optimize()
    elif optimizer == "snopt":
        snopt_optimizer.set_initial_guess(
            a=5 * np.random.rand(num_joints * num_fourier_terms),
            b=5 * np.random.rand(num_joints * num_fourier_terms),
            q0=np.random.rand(num_joints),
        )
        snopt_optimizer.optimize()
    else:
        a, b, q0 = black_box_optimizer.optimize()
        snopt_optimizer.set_initial_guess(a, b, q0)
        snopt_optimizer.optimize()


if __name__ == "__main__":
    main()
