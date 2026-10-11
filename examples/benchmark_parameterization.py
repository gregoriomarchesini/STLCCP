"""Compare rebuilt and parameterized CCP on the ReachAvoid benchmark.

Run from the repository root:
    python -m examples.benchmark_parameterization --solver ECOS --repeats 3
"""

import argparse
import contextlib
import io
import statistics
import time

import cvxpy as cp
import numpy as np

from STLCCP.benchmarks.reach_avoid import ReachAvoid
from STLCCP.solver import CCPSTLSolver


def run(horizon, seed, backend, parameterize):
    np.random.seed(seed)
    scenario = ReachAvoid((7, 8, 8, 9), (3, 5, 4, 6), horizon)
    spec = scenario.GetSpecification()
    spec.simplify()
    solver = CCPSTLSolver(
        spec, scenario.GetSystem(), np.array([1., 2., 0., 0.]),
        horizon, 10., "lse", False,
    )
    solver.AddControlBounds(np.array([-.5, -.5]), np.array([.5, .5]))
    solver.AddStateBounds(np.array([0., 0., -1., -1.]), np.array([10., 10., 1., 1.]))
    solver.AddQuadraticCost(.001 * np.diag([0., 0., 1., 1.]), .001 * np.eye(2))
    problem = cp.Problem(cp.Minimize(solver.cost), solver.constr)

    # Record the actual subproblem compilations, excluding initialization.
    original_get_data = cp.Problem.get_problem_data
    subproblems = []
    compilation_times = []

    def measured_get_data(subproblem, *args, **kwargs):
        start = time.perf_counter()
        answer = original_get_data(subproblem, *args, **kwargs)
        elapsed = time.perf_counter() - start
        if subproblem.is_qp() and any(v is solver.x for v in subproblem.variables()):
            subproblems.append(subproblem)
            compilation_times.append(elapsed)
        return answer

    cp.Problem.get_problem_data = measured_get_data
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result, elapsed = problem.solve(
                method="dccp", solver=backend, constr_penalty=solver.constr_penalty,
                weight="tree", parameterize=parameterize,
            )
    finally:
        cp.Problem.get_problem_data = original_get_data

    return {
        "elapsed": elapsed,
        "compilation": sum(compilation_times),
        "first": compilation_times[0],
        "later": statistics.mean(compilation_times[1:]) if len(compilation_times) > 1 else 0.,
        "iterations": len(subproblems),
        "models": len({id(p) for p in subproblems}),
        "objective": float(result[0]),
        "slack": float(result[1]),
        "robustness": float(np.asarray(spec.orig_robustness(solver.y.value, 0)).item()),
        "status": result[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="GUROBI")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.horizon < 25:
        parser.error("--horizon must be at least 25 for the default initial penalty")

    samples = {False: [], True: []}
    for repeat in range(args.repeats):
        # Alternate execution order to reduce a consistent first-run advantage.
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for parameterize in order:
            samples[parameterize].append(run(args.horizon, args.seed, args.solver, parameterize))

    print(f"CVXPY {cp.__version__}, solver {args.solver}, horizon {args.horizon}, seed {args.seed}")
    print(f"Median seconds over {args.repeats} runs; total includes initialization and linearization.")
    print("Mode            Total   Compilation   First compile   Later compile   Iterations   Models")
    for mode, label in ((False, "Rebuilt"), (True, "Parameterized")):
        rows = samples[mode]
        med = lambda key: statistics.median(row[key] for row in rows)
        print(f"{label:14s} {med('elapsed'):7.3f} {med('compilation'):13.3f} "
              f"{med('first'):15.3f} {med('later'):15.4f} {med('iterations'):12g} {med('models'):8g}")
        row = rows[-1]
        print(f"  {row['status']}; objective={row['objective']:.8f}; "
              f"original robustness={row['robustness']:.8f}; max slack={row['slack']:.2e}")


if __name__ == "__main__":
    main()
