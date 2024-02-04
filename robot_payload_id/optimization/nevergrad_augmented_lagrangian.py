import logging
import multiprocessing as mp

from typing import Callable, List, Tuple, Union

import nevergrad as ng
import numpy as np

from pydrake.all import (
    AugmentedLagrangianNonsmooth,
    BoundingBoxConstraint,
    MathematicalProgram,
)
from tqdm import tqdm

import wandb

# Create a cache for multiprocessing
_MAX_NUM_WORKERS = 64
_CACHED_AUGMENTED_LAGRANGIANS: List[Union[AugmentedLagrangianNonsmooth, None]] = [
    None for _ in range(_MAX_NUM_WORKERS)
]


def _multiprocessing_al_initializer(
    nonsmooth_al: Callable[[], AugmentedLagrangianNonsmooth]
):
    """Initializes the cache for multiprocessing."""
    global _CACHED_AUGMENTED_LAGRANGIANS
    _CACHED_AUGMENTED_LAGRANGIANS[
        mp.current_process()._identity[0] - 1
    ] = nonsmooth_al()


def _multiprocessing_al_cost_func(x: np.ndarray, lambda_val: np.ndarray, mu: float):
    """The augmented Lagrangian cost function process for multiprocessing."""
    global _CACHED_AUGMENTED_LAGRANGIANS
    al = _CACHED_AUGMENTED_LAGRANGIANS[mp.current_process()._identity[0] - 1]
    return al.Eval(x=x, lambda_val=lambda_val, mu=mu)


class NevergradAugmentedLagrangian:
    """
    Solve a constrained optimization program iteratively through the black-box
    optimizers in Nevergrad. We turn the constrained optimization to
    un-constrained optimization (with possible bounds on the decision
    variables) through Augmented Lagrangian (AL) approach, and then solve this
    un-constrained optimization through Nevergrad solvers. Then based on the
    constraint violation of the Nevergrad result, we then update the constraint
    penalty and the Lagrangian multipliers, and formulate a new un-constrained
    optimization problem for Nevergrad to solve again. We proceed this until
    the max iterations is reached.

    For more details, refer to section 17.4 of Numerical Optimization by Jorge Nocedal
    and Stephen Wright, Edition 1, 1999. NOTE that Drake uses μ/2 as the coefficient of
    the quadratic penalty term instead of 1/(2μ) in Edition 1.
    """

    def __init__(
        self,
        max_al_iterations: int = 10,
        budget_per_iteration: int = 3000,
        mu_multiplier: float = 2.0,
        mu_max: float = 1e3,
        method: str = "NGOpt",
        constraint_tol: float = 1e-5,
    ):
        """
        Args:
            max_al_iterations (int): The maximum number of augmented Lagrangian
                iterations.
            budget_per_iteration (int): The number of iterations to run the optimizer
                for at each augmented Lagrangian iteration.
            mu_multiplier (float): We will multiply mu by mu_multiplier if the
                equality constraint is not satisfied.
            mu_max (float): The maximum value of the penalty weights.
            method (str): The name of the Nevergrad optimizer to use. Refer to
                https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer
                for a complete list of methods.
            constraint_tol (float): If the equality constraint total violation is
                larger than this number, we will increase the penalty on the
                equality contraint violation.
        """
        self._max_al_iters = max_al_iterations
        self._budget_per_iteration = budget_per_iteration
        self._mu_multiplier = mu_multiplier
        self._mu_max = mu_max
        self._method = method
        self._constraint_tol = constraint_tol

    def _solve_al(
        self,
        nonsmooth_al: AugmentedLagrangianNonsmooth,
        x_init: np.ndarray,
        lambda_val: np.ndarray,
        mu: float,
        nevergrad_set_bounds: bool,
        pool: Union[mp.Pool, None],
        num_workers: int,
        numerically_stable_max_float: float = 1e10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solves one augmented Lagrangian iteration through Nevergrad.
        Returns a tuple of the optimal solution at this iteration, the constraint
        residue, and the augmented Lagrangian loss.
        """
        x_array = ng.p.Array(init=x_init)

        # NOTE: Settings bounds can lead to numerical instability for some optimizers
        # such as NGOpt
        all_bounds_are_inf = np.all(np.isinf(nonsmooth_al.x_lo())) and np.all(
            np.isinf(nonsmooth_al.x_up())
        )
        if nevergrad_set_bounds and not all_bounds_are_inf:
            # Replace inf with some big float that is still numerically stable
            lower_bound = np.where(
                np.isinf(nonsmooth_al.x_lo()),
                -numerically_stable_max_float,
                nonsmooth_al.x_lo(),
            )
            upper_bound = np.where(
                np.isinf(nonsmooth_al.x_up()),
                numerically_stable_max_float,
                nonsmooth_al.x_up(),
            )
            x_array.set_bounds(lower_bound, upper_bound)
            wandb.run.summary["nevergrad_set_bounds"] = True

        # Select optimizer and set initial guess
        optimizer = ng.optimizers.registry[self._method](
            parametrization=x_array,
            budget=self._budget_per_iteration,
            num_workers=num_workers,
        )
        optimizer.suggest(x_init)

        # We need to return the loss and the constraint residue. We cache these
        # terms for each suggested x value.
        x_suggests: List[np.ndarray] = []
        cached_results: List[Tuple[float, np.ndarray, float]] = []
        if num_workers > 1:
            num_iters = optimizer.budget // num_workers
            for _ in tqdm(
                range(num_iters),
                total=num_iters,
                desc="  Nevergrad Iterations",
                leave=False,
            ):
                x_lst = [optimizer.ask() for _ in range(num_workers)]
                args_list = [
                    (x_lst[i].value, lambda_val, mu) for i in range(num_workers)
                ]
                results = pool.starmap(_multiprocessing_al_cost_func, args_list)
                for i in range(num_workers):
                    al_loss, constraint_residue, cost_function_val = results[i]
                    x_suggests.append(x_lst[i].value)
                    cached_results.append(
                        (al_loss, constraint_residue, cost_function_val)
                    )
                    optimizer.tell(x_lst[i], al_loss)
        else:
            for _ in tqdm(
                range(optimizer.budget),
                total=optimizer.budget,
                desc="  Nevergrad Iterations",
                leave=False,
            ):
                x = optimizer.ask()
                al_loss, constraint_residue, cost_function_val = nonsmooth_al.Eval(
                    x=x.value, lambda_val=lambda_val, mu=mu
                )
                x_suggests.append(x.value)
                cached_results.append((al_loss, constraint_residue, cost_function_val))
                optimizer.tell(x, al_loss)

        # Get best result at this AL iteration
        recommendation = optimizer.provide_recommendation()
        # Find the entry in x_suggests closest to  recommendation.value, and take cached
        # results for that entry.
        cache_idx = np.argmin(
            np.sum((np.vstack(x_suggests) - recommendation.value) ** 2, axis=1)
        )
        al_loss, constraint_residue, cost_function_val = cached_results[cache_idx]
        wandb.log(
            {
                "Cost function value": cost_function_val,
                "augmented Lagrangian loss": al_loss,
                "mu": mu,
                "Mean absolute lambda": np.mean(np.abs(lambda_val)),
                "Max absolute lambda": np.max(np.abs(lambda_val)),
            }
        )
        wandb.run.summary["optimizer_name"] = optimizer.name
        return recommendation.value, constraint_residue, al_loss

    def compute_num_lambda(
        self, prog: MathematicalProgram, nevergrad_set_bounds: bool = True
    ) -> int:
        """Computes the number of Lagrangian multipliers in the program.

        Args:
            prog (MathematicalProgram): A MathematicalProgram object.
            nevergrad_set_bounds: If set to True, Nevergrad will call set_bounds to set
                the lower and upper bound for the decision variables; otherwise
                Nevergrad doesn't set the bounds, and the augmented Lagrangian will
                include the penalty on violating the variable bounds. It is recommended
                to set nevergrad_set_bounds to True.

        Returns:
            int: The number of Lagrangian multipliers in the program.
        """
        al = AugmentedLagrangianNonsmooth(
            prog, include_x_bounds=not nevergrad_set_bounds
        )
        return al.lagrangian_size()

    def _compute_and_log_constraint_violations(
        self,
        nonsmooth_al: AugmentedLagrangianNonsmooth,
        constraint_residue: np.ndarray,
    ) -> None:
        constraints = [c.evaluator() for c in nonsmooth_al.prog().GetAllConstraints()]
        constraint_names = [c.get_description() for c in constraints]
        constraint_types = [name.split("_")[0] for name in constraint_names]

        # Assumes that constraint name has the form "name_..."
        constraint_type_violations_map = dict(
            zip(list(set(constraint_types)), [0] * len(constraint_types))
        )

        # See Drake augmented_lagrangian.cc::EvalAugmentedLagrangian for context
        lag_idx = 0
        for constraint, constraint_type in zip(constraints, constraint_types):
            if isinstance(constraint, BoundingBoxConstraint):
                # No residuals exist for these bounds
                continue

            for i in range(constraint.num_constraints()):
                lb = constraint.lower_bound()[i]
                ub = constraint.upper_bound()[i]
                if lb == ub:
                    # Constraint adds one Lagrange multiplier
                    if constraint_residue[lag_idx] ** 2 > self._constraint_tol:
                        constraint_type_violations_map[constraint_type] += 1
                    lag_idx += 1
                else:
                    # Constraint adds 0 to 2 Lagrange multipliers
                    if not np.isinf(lb):
                        if (
                            np.maximum(-constraint_residue[lag_idx], 0) ** 2
                            > self._constraint_tol
                        ):
                            constraint_type_violations_map[constraint_type] += 1
                        lag_idx += 1

                    if not np.isinf(ub):
                        if (
                            np.maximum(-constraint_residue[lag_idx], 0) ** 2
                            > self._constraint_tol
                        ):
                            constraint_type_violations_map[constraint_type] += 1
                        lag_idx += 1

        if nonsmooth_al.include_x_bounds():
            logging.warning(
                "Skipping to log constraint violations for decision variable bounds."
            )

        num_constraint_violations = sum(list(constraint_type_violations_map.values()))
        wandb.log(
            {
                "num_constraint_violations": num_constraint_violations,
                **{
                    "constraint_violation_" + key: value
                    for key, value in constraint_type_violations_map.items()
                },
            }
        )

    def solve(
        self,
        prog_or_al_factory: Union[
            MathematicalProgram, Callable[[], AugmentedLagrangianNonsmooth]
        ],
        x_init: np.ndarray,
        lambda_val: np.ndarray,
        mu: float,
        nevergrad_set_bounds: bool = True,
        num_workers: int = 1,
        log_number_of_constraint_violations: bool = True,
        numerically_stable_max_float: float = 1e10,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Solves the constrained optimization problem through Nevergrad and augmented
        Lagrangian.

        Args:
            prog_or_al_factory: A MathematicalProgram object or a function returning a
                MathematicalProgram object.
            x_init: The initial guess of the decision variables.
            lambda_val: The initial guess of the Lagrangian multipliers.
            mu: The initial guess of the penalty weights. This must be strictly
                positive.
            nevergrad_set_bounds: If set to True, Nevergrad will call set_bounds to set
                the lower and upper bound for the decision variables; otherwise
                Nevergrad doesn't set the bounds, and the augmented Lagrangian will
                include the penalty on violating the variable bounds. It is recommended
                to set nevergrad_set_bounds to True.
            num_workers: The number of workers to use for the optimization. If one,
                then `prog_or_al_factory` is a MathematicalProgram object; otherwise,
                `prog_or_al_factory` is a pickable function that returns the
                augmented Lagrangian object.
            log_number_of_constraint_violations: If set to True, the number of
                constraint violations will be computed and logged.
            numerically_stable_max_float: A numerically stable maximum float value. This
                is used to replace inf in the bounds of the decision variables. Ideally,
                this is a bit bigger than the largest value of interest but as small
                as possible to improve the numerics.

        Returns:
            Tuple[np.ndarray, float, np.ndarray]: The optimal solution, the augmented
                Lagrangian loss, and the constraint residue.
        """
        # Validate the input
        assert num_workers > 0, "num_workers must be strictly positive."
        assert not (
            num_workers > 1 and isinstance(prog_or_al_factory, MathematicalProgram)
        ), "num_workers must be 1 if prog_or_al_factory is a MathematicalProgram object."
        assert not (
            num_workers == 1 and callable(prog_or_al_factory)
        ), "num_workers must be greater than 1 if prog_or_al_factory is a function."
        assert mu > 0, "Invalid mu! mu should be strictly positive."

        # Construct the augmented Lagrangian
        if num_workers == 1:
            nonsmooth_al = AugmentedLagrangianNonsmooth(
                prog_or_al_factory, include_x_bounds=not nevergrad_set_bounds
            )
            lagrangian_size = nonsmooth_al.lagrangian_size()
        else:
            nonsmooth_al = prog_or_al_factory()
            lagrangian_size = nonsmooth_al.lagrangian_size()
        assert lambda_val.shape[0] == lagrangian_size, "Invalid lambda_val size!"

        is_equality = nonsmooth_al.is_equality()
        x_val = x_init
        try:
            pool = (
                mp.Pool(
                    num_workers,
                    initializer=_multiprocessing_al_initializer,
                    initargs=[prog_or_al_factory],
                )
                if num_workers > 1
                else None
            )
            logging.info("Starting Nevergrad augmented Lagrangian optimization.")
            for i in tqdm(
                range(self._max_al_iters),
                total=self._max_al_iters,
                desc="AL Iterations",
            ):
                # Solve one augmented Lagrangian iteration
                x_val, constraint_residue, loss = self._solve_al(
                    nonsmooth_al=nonsmooth_al,
                    x_init=x_val,
                    lambda_val=lambda_val,
                    mu=mu,
                    nevergrad_set_bounds=nevergrad_set_bounds,
                    pool=pool,
                    num_workers=num_workers,
                    numerically_stable_max_float=numerically_stable_max_float,
                )

                # Update the Lagrangian multipliers
                equality_constraint_error = 0.0
                inequality_constraint_error = 0.0
                for j in range(lagrangian_size):
                    if is_equality[j]:
                        lambda_val[j] -= constraint_residue[j] * mu
                        equality_constraint_error += constraint_residue[j] ** 2
                    else:
                        lambda_val[j] = np.maximum(
                            lambda_val[j] - constraint_residue[j] * mu, 0
                        )
                        inequality_constraint_error += (
                            np.maximum(-constraint_residue[j], 0) ** 2
                        )

                # If the constraint error is large, then increase mu to place more emphasis
                # on satisfying the constraints. If the constraint error is small, then
                # the current mu is doing a good job of maintaining near-feasibility.
                # TODO: Consider decreasing mu by a fraction of mu_multiplier if the
                # constraint error is small.
                constraint_error = (
                    equality_constraint_error + inequality_constraint_error
                )
                if constraint_error > self._constraint_tol:
                    mu = np.minimum(mu * self._mu_multiplier, self._mu_max)

                # Log results
                wandb.log(
                    {
                        "AL Iteration": i + 1,
                        "constraint_error": constraint_error,
                        "equality_constraint_error": equality_constraint_error,
                        "inequality_constraint_error": inequality_constraint_error,
                    }
                )
                # Compute and log number of constraint violations
                if log_number_of_constraint_violations:
                    self._compute_and_log_constraint_violations(
                        nonsmooth_al=nonsmooth_al,
                        constraint_residue=constraint_residue,
                    )

        finally:
            if pool is not None:
                pool.close()
                pool.join()
                logging.info("Cleaned up pool.")

        return x_val, loss, constraint_residue
