from typing import List, Tuple

import nevergrad as ng
import numpy as np

from pydrake.all import AugmentedLagrangianNonsmooth, MathematicalProgram
from tqdm import tqdm

import wandb


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
        max_al_iterations: int = 1000,
        budget_per_iteration: int = 1500,
        mu_multiplier: float = 2.0,
        mu_max: float = 1e5,
        method: str = "OnePlusOne",
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
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solves one augmented Lagrangian iteration through Nevergrad.
        Returns a tuple of the optimal solution at this iteration, the constraint
        residue, and the augmented Lagrangian loss.
        """
        x_array = ng.p.Array(init=x_init)
        if nevergrad_set_bounds:
            x_array.set_bounds(nonsmooth_al.x_lo(), nonsmooth_al.x_up())

        # Select optimizer and set initial guess
        optimizer = ng.optimizers.registry[self._method](
            parametrization=x_array, budget=self._budget_per_iteration, num_workers=1
        )
        optimizer.suggest(x_init)

        # We need to return the loss and the constraint residue. We cache these
        # terms for each suggested x value.
        x_suggests: List[np.ndarray] = []
        cached_results: List[Tuple[float, np.ndarray, float]] = []
        for _ in tqdm(
            range(optimizer.budget),
            total=optimizer.budget,
            desc="  Nevergrad Iterations",
            leave=False,
        ):
            x = optimizer.ask()
            # TODO: NGOpt with a big budget might return NaN (same result as when
            # calling) optimizer.ask() twice. Why does this happen?
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

    def solve(
        self,
        prog: MathematicalProgram,
        x_init: np.ndarray,
        lambda_val: np.ndarray,
        mu: float,
        nevergrad_set_bounds: bool = True,
        log_number_of_constraint_violations: bool = True,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Solves the constrained optimization problem through Nevergrad and augmented
        Lagrangian.

        Args:
            prog: A MathematicalProgram object.
            x_init: The initial guess of the decision variables.
            lambda_val: The initial guess of the Lagrangian multipliers.
            mu: The initial guess of the penalty weights. This must be strictly
                positive.
            nevergrad_set_bounds: If set to True, Nevergrad will call set_bounds to set
                the lower and upper bound for the decision variables; otherwise
                Nevergrad doesn't set the bounds, and the augmented Lagrangian will
                include the penalty on violating the variable bounds. It is recommended
                to set nevergrad_set_bounds to True.
            log_number_of_constraint_violations: If set to True, the number of
                constraint violations will be computed and logged.

        Returns:
            Tuple[np.ndarray, float, np.ndarray]: The optimal solution, the augmented
                Lagrangian loss, and the constraint residue.
        """
        # Construct the augmented Lagrangian
        nonsmooth_al = AugmentedLagrangianNonsmooth(
            prog, include_x_bounds=not nevergrad_set_bounds
        )
        lagrangian_size = nonsmooth_al.lagrangian_size()

        # Validate the input
        assert lambda_val.shape[0] == lagrangian_size, "Invalid lambda_val size!"
        assert mu > 0, "Invalid mu! mu should be strictly positive."

        is_equality = nonsmooth_al.is_equality()
        x_val = x_init
        for i in tqdm(
            range(self._max_al_iters), total=self._max_al_iters, desc="AL Iterations"
        ):
            # Solve one augmented Lagrangian iteration
            x_val, constraint_residue, loss = self._solve_al(
                nonsmooth_al, x_val, lambda_val, mu, nevergrad_set_bounds
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
            constraint_error = equality_constraint_error + inequality_constraint_error
            if constraint_error > self._constraint_tol * lagrangian_size:
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
                num_constraint_violations = 0
                for j in range(lagrangian_size):
                    if is_equality[j]:
                        if np.abs(constraint_residue[j]) > self._constraint_tol:
                            num_constraint_violations += 1
                    else:
                        if constraint_residue[j] < -self._constraint_tol:
                            num_constraint_violations += 1
                wandb.log({"num_constraint_violations": num_constraint_violations})

        return x_val, loss, constraint_residue
