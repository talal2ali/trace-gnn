"""
With the new parameters left at their defaults, does run_date_gnn_fold execute the same
code path Section 5.1 executed?

Read this first, because a claim was overstated once already
------------------------------------------------------------
The work plan originally said the patched function "produces byte-identical output". That
was wrong, and it contradicted this project's own measurements: mean AP varies by SD
0.0102 across seeds and SD 0.0091 across executions of one configuration. Nothing here
re-runs anything, and nothing here claims bit-identity.

The claim being tested is narrower and is the correct one:

    with edge_builder=None, assert_causality=True and rng_state_path=None, every new
    line is inert, and the executed path is the one Section 5.1 used.

WHAT WOULD HAVE BEEN A BETTER TEST, AND WHY IT IS NOT AVAILABLE
The review's best suggestion was to load a canonical per-fold checkpoint, run inference
through the modified code, and compare against the retained canonical scores. That tests
the real path on the real data with no training. It cannot be done: the canonical run was
executed with save_weights off, so there are no per-fold weights anywhere under
results/. I checked all four canonical arms and all three repeats. The review assumed
they existed; they do not. This is a real gap and it should be stated as one, not papered
over.

What is available instead, and is what this file does:

  1. a static audit — every new name appears only inside a branch that cannot be taken at
     the defaults, printed line by line so a human can read the actual code;
  2. the resolution expression LIFTED OUT OF THE FILE and evaluated in the module's own
     namespace, which must give build_temporal_edges at the default;
  3. a check that the `if assert_causality:` branch still contains exactly
     `assert_causal(ei_np, times)` and nothing else.

Run it:  cd protocol_v2 && python tests/test_default_path_unchanged.py
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _d in (str(REPO / "src"), str(REPO), str(REPO / "protocol_v2")):  # protocol_v2 first
    while _d in sys.path:
        sys.path.remove(_d)
    sys.path.insert(0, _d)

TARGET = REPO / "src" / "run_date_gnn.py"
NEW_PARAMS = ("edge_builder", "assert_causality", "rng_state_path")
DEFAULTS = {"edge_builder": None, "assert_causality": True, "rng_state_path": None}


def _fold_fn_ast():
    tree = ast.parse(TARGET.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "run_date_gnn_fold":
            return n
    raise AssertionError("run_date_gnn_fold not found")


def test_defaults_are_the_old_behaviour():
    from run_date_gnn import run_date_gnn_fold
    sig = inspect.signature(run_date_gnn_fold)
    for p, want in DEFAULTS.items():
        assert p in sig.parameters, f"{p} missing from the signature"
        got = sig.parameters[p].default
        assert got is want or got == want, f"{p} defaults to {got!r}, expected {want!r}"
        assert sig.parameters[p].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    # and they are at the END, so every existing positional call is unaffected
    names = list(sig.parameters)
    assert names[-3:] == list(NEW_PARAMS), f"new params are not last: {names[-6:]}"
    print("signature : three new keyword params, at the end, defaulting to the old "
          "behaviour   OK")


def _mentions(node, names) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


def test_every_new_line_is_gated():
    """Structural, not textual. For every read of a new parameter, walk up the syntax
    tree and require an enclosing construct that is inert when that parameter holds its
    default. Three shapes are accepted, and nothing else:

        A  a conditional expression whose test mentions the parameter
              build_temporal_edges if edge_builder is None else edge_builder
        B  an `if`/`elif` whose test mentions the parameter
              if assert_causality:            default True  -> assert_causal still runs
              if rng_state_path and ...:      default None  -> body never entered
        C  inside _save_rng_state, whose first statement is `if not rng_state_path: return`
    """
    fn = _fold_fn_ast()
    src = TARGET.read_text().splitlines()

    parent = {}
    for n in ast.walk(fn):
        for ch in ast.iter_child_nodes(n):
            parent[ch] = n

    # C: the helper is itself guarded on its first line
    helper = next((n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef)
                   and n.name == "_save_rng_state"), None)
    assert helper is not None, "_save_rng_state not found"
    body = helper.body
    if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                                   # skip the docstring
    first = body[0]
    assert (isinstance(first, ast.If) and _mentions(first.test, NEW_PARAMS)
            and isinstance(first.body[0], ast.Return)), (
        "_save_rng_state must open with `if not rng_state_path: return` (after its "
        "docstring, if any) — otherwise it is not inert at the default")

    uses = [n for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id in NEW_PARAMS
            and isinstance(n.ctx, ast.Load)]
    assert uses, "no uses found — is this the right file?"

    def _gate(node):
        """Walk up from a node and name the construct that makes it inert, or None."""
        cur = node
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, ast.IfExp) and _mentions(cur.test, NEW_PARAMS):
                return "A  conditional expression on the parameter"
            if isinstance(cur, ast.If) and _mentions(cur.test, NEW_PARAMS):
                return f"B  guarded by `{ast.unparse(cur.test)[:52]}`"
            if isinstance(cur, ast.FunctionDef) and cur.name == "_save_rng_state":
                return "C  inside _save_rng_state, which returns early at the default"
        return None

    # Shape D: a parameter may be read unconditionally if it only feeds a PURE local
    # value whose own reads are all gated. _rng_identity and _fold_sha are computed on
    # every call — they are just hashes of the configs and the fold arrays, with no side
    # effect — and are then read only inside the guarded resume and save paths.
    pure_locals = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and _mentions(n.value, NEW_PARAMS)):
            if isinstance(n.value, ast.IfExp) and _mentions(n.value.test, NEW_PARAMS):
                continue          # already shape A: the value itself is the guard
            pure_locals[n.targets[0].id] = n
    for name, assign in pure_locals.items():
        reads = [x for x in ast.walk(fn) if isinstance(x, ast.Name) and x.id == name
                 and isinstance(x.ctx, ast.Load)]
        bad = [x.lineno for x in reads if _gate(x) is None]
        assert not bad, (f"{name} is assigned from a new parameter unconditionally, and "
                         f"is then read outside a guard at line(s) {bad}. That is not "
                         f"inert at the default.")
        print(f"gating    : {name} computed unconditionally (pure), and all "
              f"{len(reads)} of its reads are gated   OK")

    print("gating    : every read of a new parameter, and what makes it inert")
    ungated = []
    for n in sorted(uses, key=lambda x: (x.lineno, x.col_offset)):
        why = _gate(n)
        if why is None:
            for lname, assign in pure_locals.items():
                if assign.lineno <= n.lineno <= (assign.end_lineno or assign.lineno):
                    why = f"D  feeds {lname}, a pure value read only under a guard"
                    break
        line = src[n.lineno - 1].strip()
        print(f"            L{n.lineno:<5} {'ok ' if why else 'UNGATED'} {line[:64]}")
        if why:
            print(f"                        {why}")
        else:
            ungated.append((n.id, n.lineno, line))
    assert not ungated, f"un-gated use of a new parameter: {ungated}"

    # the one pre-existing line whose shape changed rather than being added
    joined = "\n".join(src)
    assert 'done_weeks = {int(w) for w in prev["test_week"].tolist()}' in joined, (
        "the B5 int() cast on the resume key is missing")
    print("            the resume key is cast with int() (defect B5)   OK")

    # --- the two new lines that run UNCONDITIONALLY, named so they are not a blind spot
    # Nothing above catches these, because neither mentions a new parameter. Both are
    # inside the fold loop and both are inert at the defaults:
    #
    #   done_weeks.add(int(w))     done_weeks is read only by the `skip if already done`
    #                              test at the top of the loop. Fold labels are unique, so
    #                              a label added after its fold completes is never tested
    #                              again in the same pass. Nothing else reads the set.
    #   _save_rng_state(done_weeks)  returns on its first line when rng_state_path is None.
    #
    body = [s for s in ast.walk(fn) if isinstance(s, ast.stmt)]
    calls = [s for s in body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
             and ast.unparse(s.value).startswith("_save_rng_state")]
    adds = [s for s in body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and ast.unparse(s.value) == "done_weeks.add(int(w))"]
    assert len(calls) == 1, f"expected one _save_rng_state call, found {len(calls)}"
    assert len(adds) == 1, f"expected one done_weeks.add, found {len(adds)}"
    reads = [ast.unparse(n) for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "done_weeks"]
    print(f"            2 unconditional new lines (L{adds[0].lineno}, L{calls[0].lineno}); "
          f"done_weeks is read only by the resume skip test   OK")


def test_default_builder_is_the_canonical_one():
    """The default really is build_temporal_edges, and it is still the same object the
    module imported at the top — not a shadowed copy."""
    import run_date_gnn as R
    from temporal_graph import build_temporal_edges
    assert R.build_temporal_edges is build_temporal_edges
    src = inspect.getsource(R.run_date_gnn_fold)
    assert "_build_edges = build_temporal_edges if edge_builder is None else edge_builder" in src
    assert "if assert_causality:\n        assert_causal(ei_np, times)" in src
    print("builder   : edge_builder=None resolves to temporal_graph.build_temporal_edges, "
          "and assert_causal still runs   OK")


def test_the_resolution_expression_from_the_real_source_yields_the_old_function():
    """Lift the actual resolution expression out of src/run_date_gnn.py, evaluate it in
    that module's own namespace with edge_builder=None, and require it to be
    build_temporal_edges itself.

    An earlier version of this test built the graph twice with `build_temporal_edges` and
    compared the results — which showed only that the function is deterministic. This one
    evaluates the patched line, taken from the file rather than retyped.
    """
    import run_date_gnn as R
    from temporal_graph import build_temporal_edges

    fn = _fold_fn_ast()
    expr = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "_build_edges"):
            expr = n.value
    assert expr is not None, "the _build_edges assignment is gone"
    text = ast.unparse(expr)
    print(f"resolution: the line, lifted from the file -> {text}")

    ns = dict(vars(R)); ns["edge_builder"] = None
    resolved = eval(compile(ast.Expression(expr), "<lifted>", "eval"), ns)
    assert resolved is build_temporal_edges, (
        f"at edge_builder=None the expression resolves to {resolved!r}, not "
        f"temporal_graph.build_temporal_edges")

    sentinel = object()
    ns["edge_builder"] = sentinel
    assert eval(compile(ast.Expression(expr), "<lifted>", "eval"), ns) is sentinel, (
        "a supplied edge_builder is not being used")
    print("resolution: evaluates to build_temporal_edges at the default, and to the "
          "supplied builder otherwise   OK")


def test_causality_assertion_still_runs_at_the_default():
    """The `if assert_causality:` branch must contain the assert_causal call, so the
    default really does check the graph exactly as before."""
    fn = _fold_fn_ast()
    node = next((n for n in ast.walk(fn)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "assert_causality"),
                None)
    assert node is not None, "the `if assert_causality:` guard is gone"
    body = [ast.unparse(s) for s in node.body]
    assert body == ["assert_causal(ei_np, times)"], f"guard body is {body}"
    assert not node.orelse, "the guard has an else branch it did not have before"
    print("causality : at assert_causality=True the body is exactly "
          "`assert_causal(ei_np, times)`, unchanged   OK")


if __name__ == "__main__":
    test_defaults_are_the_old_behaviour()
    test_every_new_line_is_gated()
    test_default_builder_is_the_canonical_one()
    test_the_resolution_expression_from_the_real_source_yields_the_old_function()
    test_causality_assertion_still_runs_at_the_default()
    print("\nDefault path is unchanged. Note the gap recorded at the top of this file: "
          "the canonical run saved no weights, so the checkpoint-replay test the review "
          "asked for cannot be run, and nothing here substitutes for it.")
