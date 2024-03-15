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
`.venv/bin/activate` and `.venv/bin/activate.nu` files, modifying the paths and python
version as required:
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

*Hint:* Run on multiple cores using `--num_workers`. When using multiple workers,
using `--log_level ERROR` is needed for nice progress bars.

*Note:* It is recommended to design trajectories without considering reflected inertia
and joint friction as this seems to lead to better results, even when identifying these
parameters later on.

The `--payload_only` flag enables designing trajectories that only optimize the
excitation of the payload parameters. These are the 10 inertial parameters of the last
link.

### Use a Fourier series trajectory as an initial guess for BSpline trajectory optimization

First, convert the optimized Fourier series trajectory into a BSpline trajectory:
```bash
python scripts/create_bspline_traj_from_fourier_series.py \
--traj_parameter_path logs/fourier_series_traj \
--save_dir logs/converted_trajs/bspline_traj \
--num_control_points_initial 30 --num_timesteps 1000
```

Second, use the converted trajectory as the initial guess:
```bash
python scripts/design_optimal_excitation_trajectories.py  \
--optimizer "black_box" --cost_function "condition_number_and_e_optimality" \
--num_timesteps 1000 --use_one_link_arm --logging_path logs/traj_bspline \
--traj_initial logs/converted_trajs/bspline_traj --use_bspline \
--num_control_points 30
```

*Note:* When using multiple workers for BSpline optimization, it is best to use `CMAstd`
as the optimizer. The default optimizers seem to have bugs (see
[issue](https://github.com/facebookresearch/nevergrad/issues/1593)).

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

## Collect joint data

```bash
python scripts/collect_joint_data.py --scenario_path models/iiwa_scenario.yaml \
--traj_parameter_path logs/traj_bspline --save_data_path joint_data/iiwa
```

Add the `--use_hardware` flag to collect data on the real robot.

## Process collected joint data

The collected joint data will likely be quite noisy.

It can help to average joint data from executing the same trajectory multiple times
for improving the signal-to-noise ratio:
```bash
python scripts/average_joint_data.py joint_data_dir/ joint_data_averaged/
```
where `joint_data_dir` contains the joint data directories to average and
`joint_data_averaged` is the directory to write the averaged joint data to.

Filtering is very important and it is recommended to tune the parameters carefully.
Sweeping over different filter parameters can be helpful in this regard (see
sweeping section below).
Once filtering parameters have been determined, the data can be processed using the
`scripts/process_joint_data.py` script or by passing the parameters as arguments to
`scripts/solve_inertial_param_sdp.py` with the `--process_joint_data` flag.

After filtering, one might want to increase the data amount by combining the data from
multiple excitation trajectories. This can be achieved using
`scripts/concatenate_joint_data.py`.

## SDP System ID

Generates data, constructs the data matrix and solves the SDP using posidefinite
constraints on the pseudo inertias.
This requires trajectories that have been designed using optimal excitation trajectory
design as otherwise the numerics won't be good enough for the optimization to succeed.

Generating GT/ model-predicted data:
```bash
python scripts/solve_inertial_param_sdp.py --traj_parameter_path logs/traj \
--num_data_points 5000 --use_one_link_arm
```
Note that the model-predicted data corresponds to simulation data from
`collect_joint_data.py` if the simulation timestep is set to zero (continuous-time
simulation). Otherwise, the simulation data will be slightly noisy and closer to real
robot data.

Using collected data (sim or real):
```bash
python scripts/solve_inertial_param_sdp.py --joint_data_path joint_data/iiwa_only \
--process_joint_data
```

### Identifying the arm parameters and then freeze the parameters to identify the payload

First, identify the arm parameters without payload and save them to disk:
```bash
python scripts/solve_inertial_param_sdp.py --joint_data_path joint_data/iiwa_only \
--process_joint_data --output_param_path identified_params/params.npy
```

Second, freeze the identified parameters and identify the payload:
```bash
python scripts/solve_inertial_param_sdp.py \
--joint_data_path joint_data/iiwa_with_payload --process_joint_data \
--initial_param_path identified_params/params.npy --payload_only
```
The payload inertial parameters should correspond to the last link parameters
identified by the second run minus the ones identified by the first run, i.e. the ones
stored in `identified_params/params.npy` and passed to the second run. This parameter
difference is printed by the script. Specify `--payload_frame_name` if you want to
print them in a particular frame.

## Reparameterized System ID

```bash
python scripts/identify_model.py --config-name iiwa_id
```

## Sweeping Parameters

### Eric ID

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

### SDP data processing

The SDP results are very sensitive to the data processing. It can make sense to
sweep over the data processing parameters to identify the best parameters for ones
particular collected joint data.

A sweep can be started with
```bash
wandb sweep config/sweep/sdp_data_sweep.yaml
```
Individual agents for the sweep can be started using the printed `wandb agent` command.

## Evaluation

Inertial ellipsoids can be visualized with `scripts/visualize_inertial_ellipsoids.py`.

## Credit

Any code in `robot_payload_id/eric_id` has been copied/ adopted from Eric Cousineau
([Github repo](https://github.com/EricCousineau-TRI/drake_sys_id)).
