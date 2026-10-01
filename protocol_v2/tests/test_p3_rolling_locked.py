"""
Can P3_rolling be retrained or overwritten? This is verification item (a), and this file
is the automated form of it. Everything it checks is also readable by eye in guards.py.

It checks all four locks, plus the one that matters most in practice: that no notebook
anywhere in protocol_v2 asks for that arm, and that the training function is called from
exactly one place.

Run it:  cd protocol_v2 && python tests/test_p3_rolling_locked.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _d in (str(REPO / "src"), str(REPO), str(REPO / "protocol_v2")):  # protocol_v2 first
    while _d in sys.path:
        sys.path.remove(_d)
    sys.path.insert(0, _d)

import config as C            # noqa: E402
import guards                 # noqa: E402

PV2 = Path(C.PROTOCOL_V2)

# Folders holding read-only COPIES for review, not code protocol_v2 runs. Both contain a
# copy of src/run_date_gnn.py, which defines run_date_gnn_fold — so without this exclusion
# the "exactly one call site" check counts the copies and fails.
REVIEW_COPY_DIRS = ("_canonical_reference", "_changed_src_for_review")


def _is_review_copy(path: Path) -> bool:
    rel = path.relative_to(PV2) if path.is_absolute() else Path(path)
    return any(part in REVIEW_COPY_DIRS for part in rel.parts)


# ------------------------------------------------------------------ 1. no specification
def test_p3_rolling_has_no_training_spec():
    assert "P3_rolling" in C.LOCKED_ARMS
    assert "P3_rolling" not in C.TRAINABLE_ARMS
    assert "P3_rolling" not in C.ARM_SPEC, (
        "P3_rolling has a training specification. It must have none — there should be "
        "nothing for a trainer to consume.")
    assert set(C.ARM_SPEC) == set(C.TRAINABLE_ARMS) == {
        "P2_cutoff", "P1_causal", "P1_leaky", "P3_acausal"}
    print("1 spec    : P3_rolling has no fold set, no graph choice, no output dir   OK")


# ------------------------------------------------------------------ 2. one guarded call
def test_training_function_is_called_exactly_once():
    hits = []
    for p in sorted(PV2.rglob("*.py")):
        if _is_review_copy(p):
            continue                     # read-only review copies, not live code
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\brun_date_gnn_fold\s*\(", line):
                hits.append((p.relative_to(PV2), i, line.strip()))
    calls = [h for h in hits if "tests/" not in str(h[0])]
    assert len(calls) == 1, f"expected one call site, found {len(calls)}: {calls}"
    assert calls[0][0] == Path("runner.py"), f"call site is {calls[0][0]}, not runner.py"

    # and its enclosing function opens with the guard
    src = (PV2 / "runner.py").read_text()
    fn = src.split("def train_arm(")[1]
    body = fn.split('"""')[2]
    first = next(l.strip() for l in body.splitlines() if l.strip())
    assert first.startswith("guards.assert_trainable(arm)"), (
        f"train_arm does not open with the guard; it opens with: {first}")
    print(f"2 callsite: one call to run_date_gnn_fold, in runner.py L{calls[0][1]}, "
          f"inside train_arm, whose first statement is the guard   OK")


def _identifiers(src: str):
    """Names actually referenced by a code cell. Parsed, so prose in comments and
    docstrings — of which these notebooks have plenty — does not count as a reference."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), src
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.alias):
            names.add(n.name.split(".")[0]); names.add((n.asname or "").strip())
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module.split(".")[0])
    return names, ast.unparse(tree)


def test_notebooks_never_request_the_locked_arm():
    """No notebook may pass P3_rolling to anything that trains, may import the training
    function directly, or may touch the deprecated runner."""
    bad = []
    for nb in sorted((PV2 / "notebooks").glob("*.ipynb")):
        if _is_review_copy(nb):
            continue
        doc = json.loads(nb.read_text())
        for ci, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            names, unparsed = _identifiers("".join(cell["source"]))
            if "run_date_gnn_fold" in names:
                bad.append(f"{nb.name} cell {ci}: references run_date_gnn_fold directly")
            if "protocol_runner" in names or "run_protocol_fold" in names:
                bad.append(f"{nb.name} cell {ci}: uses the deprecated protocol_runner")
            for m in re.finditer(r"train_arm\(\s*([\"'])(\w+)\1", unparsed):
                if m.group(2) in C.LOCKED_ARMS:
                    bad.append(f"{nb.name} cell {ci}: train_arm({m.group(2)!r})")
            for locked in C.LOCKED_ARMS:
                if re.search(rf"train_arm\([^)]*{locked}", unparsed):
                    bad.append(f"{nb.name} cell {ci}: {locked} reaches a train_arm call")
    assert not bad, "notebooks reach for the locked arm or the deprecated runner:\n  " + \
                    "\n  ".join(bad)
    print("2 notebooks: no notebook trains P3_rolling, references the training function "
          "directly, or touches protocol_runner   OK")


def test_assert_trainable_refuses():
    for arm in C.LOCKED_ARMS:
        try:
            guards.assert_trainable(arm)
        except guards.LockedArmError as e:
            assert "LOCKED" in str(e) and "never be trained" in str(e)
        else:
            raise AssertionError(f"assert_trainable({arm!r}) did not raise")
    for arm in C.TRAINABLE_ARMS:
        assert guards.assert_trainable(arm) == arm
    try:
        guards.assert_trainable("P3_Rolling")           # a typo must not slip through
    except guards.LockedArmError:
        pass
    else:
        raise AssertionError("an unknown arm name was accepted")
    print("2 guard   : assert_trainable refuses P3_rolling and any unknown name   OK")


# ------------------------------------------------------------------ 3. writes are fenced
def test_writes_outside_protocol_v2_are_refused():
    import runner
    forbidden = [
        C.CANONICAL_P3_DIR / "per_fold.csv",
        C.CANONICAL_P3_DIR / "scores.csv",
        C.CANONICAL_RUN / "run_manifest.json",
        C.REPO / "results" / "protocol" / "per_week_all_arms.csv",
        C.REPO / "notebooks" / "08b_protocol_11feat.ipynb",
        C.REPO / "paper" / "TRACE-GNN_manuscript_v6b.docx",
        C.REPO / "src" / "run_date_gnn.py",
    ]
    for p in forbidden:
        try:
            guards.assert_writable(p)
        except guards.LockedArmError:
            continue
        raise AssertionError(f"assert_writable permitted {p}")
    # and the permitted root really is permitted
    guards.assert_writable(C.ARMS_DIR / "P1_causal" / "scores.csv")
    # arm_dir refuses the locked arm outright
    try:
        runner.arm_dir("P3_rolling")
    except guards.LockedArmError:
        pass
    else:
        raise AssertionError("arm_dir('P3_rolling') returned a path")
    print(f"3 fence   : {len(forbidden)} paths outside protocol_v2/results refused, "
          f"including every canonical file and the manuscript   OK")


# ------------------------------------------------------------------ 4. the checksum lock
def test_checksum_lock_detects_a_change():
    """Copy the canonical directory to a scratch location, lock it, change one byte, and
    confirm the comparison fails. The real directory is never written to."""
    src = Path(C.CANONICAL_P3_DIR)
    if not src.is_dir():
        print("4 checksum: canonical directory not present here, skipped")
        return
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "main"
        shutil.copytree(src, fake)
        before = guards.checksum_tree(fake)
        assert before["n_files"] >= 5
        target = fake / "per_fold.csv"
        target.write_bytes(target.read_bytes() + b"\n")     # one byte
        after = guards.checksum_tree(fake)
        assert after["combined_sha256"] != before["combined_sha256"]
        # an ADDED file must also move the digest, not just an edited one
        (fake / "sneaky.csv").write_text("x")
        assert guards.checksum_tree(fake)["combined_sha256"] != after["combined_sha256"]
    print(f"4 checksum: a one-byte edit and an added file both change the combined "
          f"digest ({before['n_files']} files)   OK")


def test_lockfile_records_the_real_canonical_directory():
    rec = guards.lock_canonical()
    assert rec["arm"] == "P3_rolling"
    assert Path(rec["root"]).resolve() == Path(C.CANONICAL_P3_DIR).resolve()
    assert rec["n_files"] >= 5
    guards.assert_canonical_unchanged(verbose=False)
    print(f"4 lockfile: {C.CANONICAL_LOCKFILE.name} covers {rec['n_files']} files, "
          f"combined {rec['combined_sha256'][:16]}...   OK")


# --------------------------------------------------------------- naming, kept explicit
def test_the_two_p3_names_are_never_conflated():
    assert "P3_acausal" in C.TRAINABLE_ARMS and "P3_rolling" in C.LOCKED_ARMS
    assert C.ARM_SPEC["P3_acausal"]["edges"] == "acausal"
    import assembly_settings as A
    assert A.FACTORIAL_2X2[("rolling origin", "causal graph")] == "P3_rolling"
    assert A.FACTORIAL_2X2[("rolling origin", "unconstrained graph")] == "P3_acausal"
    print("naming    : P3_acausal is retrained, P3_rolling is not; the 2x2 keeps them "
          "on separate cells   OK")


if __name__ == "__main__":
    test_p3_rolling_has_no_training_spec()
    test_training_function_is_called_exactly_once()
    test_notebooks_never_request_the_locked_arm()
    test_assert_trainable_refuses()
    test_writes_outside_protocol_v2_are_refused()
    test_checksum_lock_detects_a_change()
    test_lockfile_records_the_real_canonical_directory()
    test_the_two_p3_names_are_never_conflated()
    print("\nP3_rolling cannot be trained or overwritten by this code.")
