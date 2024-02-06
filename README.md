# robot_payload_id
Identifying object inertial properties using a robot arm

## Installation

This repo uses Poetry for dependency management. To setup this project, first install
[Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10
installed on your system.

Then, configure poetry to setup a virtual environment that uses Python 3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the
following command:
```
poetry install -vvv
```
(the `-vvv` flag adds verbose output).

For local Drake and manipulation installations, insert the following at the end of the
`.venv/bin/activate` and `.venv/bin/activate.nu` files, modifying the paths and python version as required:
```bash
export PYTHONPATH=~/drake-build/install/lib/python3.10/site-packages:${PYTHONPATH}
export PYTHONPATH=~/manipulation:${PYTHONPATH}
```

Activate the environment:
```
poetry shell
```

Install `git-lfs`:

```bash
git-lfs install
git-lfs pull
```

## Optimal Experiment Design

```bash
python scripts/design_optimal_excitation_trajectories.py  \
--optimizer "black_box" --cost_function "condition_number_and_e_optimality" \
--num_fourier_terms 5 --num_timesteps 1000 --use_one_link_arm --logging_path logs/traj
```

### Use a Fourier series trajectory as an initial guess for BSpline trajectory optimization

First, convert the optimized Fourier series trajectory into a BSpline trajectory:
```bash
python scripts/create_bspline_traj_from_fourier_series.py \
--traj_parameter_path logs/fourier_series_traj \
--save_dir logs/converted_trajs/bspline_traj \
--num_control_points_initial 30 --num_timesteps 30
```

Second, use the converted trajectory as the initial guess:
```bash
python scripts/design_optimal_excitation_trajectories.py  \
--optimizer "black_box" --cost_function "condition_number_and_e_optimality" \
--num_timesteps 1000 --use_one_link_arm --logging_path logs/traj_bspline \
--traj_initial logs/converted_trajs/bspline_traj --use_bspline \
--num_control_points 30
```

### Visualize the designed trajectories

Make sure to use the same parameters for `num_timesteps` and `time_horizon` as were used
for the optimal trajectory design.
```bash
python scripts/visualize_trajectory.py --traj_parameter_path logs/traj
```

## Symbolic System ID

```bash
python scripts/symbolic_id.py --config-name one_link_arm_symbolic_id
```

## SDP System ID

Generates data, constructs the data matrix and solves the SDP using posidefinite
constraints on the pseudo inertias.
This requires trajectories that have been designed using optimal excitation trajectory
design as otherwise the numerics won't be good enough for the optimization to succeed.

```bash
python scripts/solve_inertial_param_sdp.py --use_one_link_arm \
--remove_unidentifiable_params --traj_parameter_path logs/traj
```

NOTE that one would want to obtain data using optimal experiment design to ensure that
the numerics are good enough (e.g. condition number optimization).

## Reparameterized System ID

```bash
python scripts/identify_model.py --config-name iiwa_id
```

### Sweeping Parameters

A sweep can be started with
```bash
wandb sweep config/sweep/iiwa_id_sweep.yaml
```
Individual agents for the sweep can be started using the printed `wandb agent` command.

A parallel sweep can be started with
```bash
bash scripts/parallel_sweep.sh config/sweep/iiwa_id_sweep.yaml ${NUM_PARALLEL}
```
where `NUM_PARALLEL` is a variable containing the number of parallel runs. By default,
the maximum number of cores is used.
