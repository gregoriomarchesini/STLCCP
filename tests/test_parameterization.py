import cvxpy as cp
import numpy as np
import pytest

from STLCCP.benchmarks.reach_avoid import ReachAvoid
from STLCCP.solver import CCPSTLSolver
from STLCCP.solver.dccp.linearize import linearize
from log_sum_exp import log_sum_exp as stable_log_sum_exp


@pytest.mark.parametrize("kind", ["scalar", "vector", "matrix", "lse", "mellowmin"])
def test_updated_tangents_match_rebuilt_tangents(kind):
    x = cp.Variable((2, 3))
    if kind == "scalar":
        expr = -cp.square(x[0, 1]) - cp.square(x[1, 2]) + 3 * x[1, 0]
    elif kind == "vector":
        expr = -cp.square(x[:, 1])
    elif kind == "matrix":
        expr = -cp.square(x)
    else:
        inputs = cp.hstack([x[0, 0], 2 * x[1, 2] + 1, x[1, 0]])
        expr = -cp.log_sum_exp(-10 * inputs) / 10
        if kind == "mellowmin":
            expr += np.log(3) / 10

    cache = {}
    rng = np.random.default_rng(42)
    tangent = None
    for expansion_point in rng.normal(size=(3, 2, 3)):
        x.value = expansion_point
        rebuilt = linearize(expr)
        updated = linearize(expr, cache=cache)
        if tangent is not None:
            assert updated is tangent
        tangent = updated
        np.testing.assert_allclose(updated.value, expr.value, atol=1e-12)
        for probe in rng.normal(size=(3, 2, 3)):
            x.value = probe
            np.testing.assert_allclose(updated.value, rebuilt.value, atol=1e-12)
    assert cp.Problem(cp.Minimize(cp.sum(tangent))).is_dcp(dpp=True)


def test_tangent_accepts_integer_initial_value():
    x = cp.Variable(1)
    x.value = np.array([2])
    expr = -cp.square(x)
    cached = linearize(expr, cache={})
    rebuilt = linearize(expr)
    x.value = np.array([3])
    np.testing.assert_allclose(cached.value, rebuilt.value, atol=1e-12)


def test_high_k_tangent_can_change_dominant_input():
    # Use the stable atom supplied by the repository for large smoothing k.
    x = cp.Variable(3)
    expr = -stable_log_sum_exp(-1000 * x) / 1000
    cache = {}
    for point in ([0., 1., 2.], [2., 1., 0.]):
        x.value = np.array(point)
        cached = linearize(expr, cache=cache)
        rebuilt = linearize(expr)
        x.value = np.array([1., 0.5, 1.])
        np.testing.assert_allclose(cached.value, rebuilt.value, atol=1e-12)


def run_ccp(problem, weights, weight_type, parameterize):
    return problem.solve(
        method="dccp", solver="ECOS", constr_penalty=weights,
        weight=weight_type, parameterize=parameterize, tau=0.2,
    )[0]


@pytest.mark.parametrize("weight_type", ["tree", "decay", "50", "smallest"])
def test_ccp_matches_rebuilt_path_and_reuses_compilation(monkeypatch, weight_type):
    original_solve = cp.Problem.solve
    records = []

    def record_solve(problem, *args, **kwargs):
        result = original_solve(problem, *args, **kwargs)
        if "method" not in kwargs and problem.parameters():
            records.append((problem, problem._cache.param_prog))
        return result

    monkeypatch.setattr(cp.Problem, "solve", record_solve)
    results = []
    for parameterize in (False, True):
        x = cp.Variable(1)
        x.value = np.array([2.])
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(x - 0.25)),
            [1 - cp.square(x) <= 0, 0.5 - cp.square(x) <= 0, x >= 0, x <= 3],
        )
        results.append(run_ccp(problem, [3, 7, 1, 1], weight_type, parameterize))
    assert results[0][-1] == results[1][-1] == "Converged"
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-7)
    np.testing.assert_allclose(results[0][2][0], results[1][2][0], atol=1e-7)
    assert results[0][1] <= 1e-5 and results[1][1] <= 1e-5
    assert len(records) >= 2
    assert all(record[0] is records[0][0] for record in records)
    assert records[0][1] is not None
    assert all(record[1] is records[0][1] for record in records)


@pytest.mark.parametrize("sense", [cp.Minimize, cp.Maximize])
def test_non_dcp_objective_matches_rebuilt_path(sense):
    results = []
    for parameterize in (False, True):
        x = cp.Variable(1)
        x.value = np.array([1.5])
        cost = -cp.square(x) if sense is cp.Minimize else cp.square(x)
        problem = cp.Problem(sense(cp.sum(cost)), [1 - cp.square(x) <= 0, x <= 3])
        results.append(run_ccp(problem, [3, 1], "tree", parameterize))
    assert results[0][-1] == results[1][-1] == "Converged"
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-6)


def test_reach_avoid_matches_rebuilt_path():
    results = []
    for parameterize in (False, True):
        np.random.seed(42)
        scenario = ReachAvoid((7, 8, 8, 9), (3, 5, 4, 6), 12)
        spec = scenario.GetSpecification()
        spec.simplify()
        solver = CCPSTLSolver(
            spec, scenario.GetSystem(), np.array([1., 2., 0., 0.]),
            12, 10., "lse", False,
        )
        solver.AddControlBounds(np.array([-.5, -.5]), np.array([.5, .5]))
        solver.AddStateBounds(np.array([0., 0., -1., -1.]), np.array([10., 10., 1., 1.]))
        solver.AddQuadraticCost(.001 * np.diag([0., 0., 1., 1.]), .001 * np.eye(2))
        problem = cp.Problem(cp.Minimize(solver.cost), solver.constr)
        result = run_ccp(problem, solver.constr_penalty, "tree", parameterize)
        results.append((result, solver.x.value, solver.u.value, spec.orig_robustness(solver.y.value, 0)))
    assert results[0][0][-1] == results[1][0][-1] == "Converged"
    assert results[0][0][1] <= 1e-5 and results[1][0][1] <= 1e-5
    np.testing.assert_allclose(results[0][0][0], results[1][0][0], atol=1e-6)
    for baseline, parameterized in zip(results[0][1:], results[1][1:]):
        np.testing.assert_allclose(baseline, parameterized, atol=1e-4)
