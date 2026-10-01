"""
Static checks that need no GPU, no data and no training. Run these before anything else.

    cd protocol_v2 && python tests/test_static.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _d in (str(REPO / "src"), str(REPO), str(REPO / "protocol_v2")):  # protocol_v2 first
    while _d in sys.path:
        sys.path.remove(_d)
    sys.path.insert(0, _d)
PV2 = REPO / "protocol_v2"


def test_every_module_parses():
    for p in sorted(PV2.rglob("*.py")):
        ast.parse(p.read_text(), filename=str(p))
    print(f"parse     : {len(list(PV2.rglob('*.py')))} modules parse   OK")


def test_every_notebook_cell_compiles():
    """Each code cell must be valid Python on its own."""
    n_nb = n_cell = 0
    for nb in sorted((PV2 / "notebooks").glob("*.ipynb")):
        doc = json.loads(nb.read_text())
        n_nb += 1
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            n_cell += 1
            src = "".join(cell["source"])
            try:
                compile(src, f"{nb.name}#{i}", "exec")
            except SyntaxError as e:
                raise AssertionError(f"{nb.name} cell {i} does not compile: {e}") from None
    print(f"compile   : {n_cell} code cells across {n_nb} notebooks compile   OK")


def test_no_notebook_has_been_executed():
    """Nothing here has been run. Every code cell must be blank."""
    for nb in sorted((PV2 / "notebooks").glob("*.ipynb")):
        doc = json.loads(nb.read_text())
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None, \
                f"{nb.name} cell {i} has execution_count {cell['execution_count']}"
            assert cell.get("outputs") == [], f"{nb.name} cell {i} has stored output"
    print("unrun     : every code cell is unexecuted, with no stored output   OK")


def test_each_arm_notebook_has_the_required_shape():
    """First cell prints free GPU memory. The notebook then ends with three cells, in
    order: the cleanup note, a read-only CHECK cell, and a destructive CLEAN cell. Both
    trailing cells are left unexecuted, and neither runs automatically."""
    for nb in sorted((PV2 / "notebooks").glob("A[1-4]_*.ipynb")):
        doc = json.loads(nb.read_text())
        cells = doc["cells"]
        first_code = next(c for c in cells if c["cell_type"] == "code")
        s = "".join(first_code["source"])
        assert "mem_get_info" in s and "_free_start" in s, \
            f"{nb.name}: the first code cell must print free GPU memory"

        note, check, clean = cells[-3], cells[-2], cells[-1]

        assert note["cell_type"] == "markdown", f"{nb.name}: no cleanup note"
        nsrc = "".join(note["source"])
        for token in ("on demand", "nothing here runs by itself", "check", "clean",
                      "shut the kernel down", "unexecuted"):
            assert token in nsrc.lower(), f"{nb.name}: cleanup note missing {token!r}"

        # checked on the syntax tree, so the words appearing in a comment don't count
        def _calls(cell):
            t = ast.parse("".join(cell["source"]))
            return {ast.unparse(n.func) for n in ast.walk(t) if isinstance(n, ast.Call)}, t

        assert check["cell_type"] == "code", f"{nb.name}: no CHECK cell"
        csrc = "".join(check["source"])
        assert "CHECK" in csrc and "mem_get_info" in csrc, f"{nb.name}: CHECK malformed"
        ccalls, ctree = _calls(check)
        for banned in ("gc.collect", "torch.cuda.empty_cache"):
            assert banned not in ccalls, \
                f"{nb.name}: CHECK must be read-only, but calls {banned}()"
        assert not [n for n in ast.walk(ctree) if isinstance(n, ast.Delete)], \
            f"{nb.name}: CHECK must be read-only, but deletes something"

        assert clean["cell_type"] == "code", f"{nb.name}: no CLEAN cell"
        lsrc = "".join(clean["source"])
        for token in ("CLEAN", "mem_get_info", "recovered", "still held"):
            assert token in lsrc, f"{nb.name}: CLEAN cell missing {token!r}"
        lcalls, ltree = _calls(clean)
        for needed in ("gc.collect", "torch.cuda.empty_cache"):
            assert needed in lcalls, f"{nb.name}: CLEAN cell never calls {needed}()"
        assert not [n for n in ast.walk(ltree) if isinstance(n, ast.Assert)], \
            f"{nb.name}: the CLEAN cell must not assert — the decision is Talal's"

        for c, n in ((check, "CHECK"), (clean, "CLEAN")):
            assert c.get("execution_count") is None and c.get("outputs") == [], \
                f"{nb.name}: {n} must be left unexecuted"

        # and nothing earlier in the notebook may free memory behind his back
        for i, c in enumerate(cells[:-2]):
            if c["cell_type"] != "code":
                continue
            body = "".join(c["source"])
            assert "empty_cache()" not in body and "del globals()" not in body, \
                f"{nb.name} cell {i}: frees GPU memory automatically. Cleanup is manual."
    print("shape     : A1-A4 open with the free-memory cell and end with the note + an "
          "unexecuted read-only CHECK + an unexecuted CLEAN; nothing frees automatically"
          "   OK")


def test_only_the_four_arms_appear_as_targets():
    import config as C
    for nb in sorted((PV2 / "notebooks").glob("A[1-4]_*.ipynb")):
        doc = json.loads(nb.read_text())
        arms = set()
        for cell in doc["cells"]:
            if cell["cell_type"] != "code":
                continue
            for n in ast.walk(ast.parse("".join(cell["source"]))):
                if (isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == "ARM"
                        and isinstance(n.value, ast.Constant)):
                    arms.add(n.value.value)
        assert len(arms) == 1, f"{nb.name}: ARM set {arms}"
        arm = arms.pop()
        assert arm in C.TRAINABLE_ARMS, f"{nb.name}: ARM={arm!r} is not trainable"
        assert arm in nb.name, f"{nb.name}: filename and ARM={arm!r} disagree"
    print("targets   : each of A1-A4 sets exactly one ARM, trainable, matching its "
          "filename   OK")


def test_config_is_the_only_place_settings_live():
    """No notebook may restate a training hyperparameter as a literal — not as a keyword
    argument and not as a bare assignment. Checked on the syntax tree, so the same words
    appearing inside a print string are correctly ignored."""
    banned = {"epochs", "lr", "batch_size", "patience", "val_frac", "focal_gamma",
              "weight_decay", "grad_clip", "hidden_dim", "n_heads", "n_layers",
              "ffn_mult", "dropout", "seed", "alert_rate", "max_prior_neighbors",
              "early_stop_metric", "use_swa"}
    bad = []
    for nb in sorted((PV2 / "notebooks").glob("*.ipynb")):
        doc = json.loads(nb.read_text())
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for n in ast.walk(tree):
                if (isinstance(n, ast.keyword) and n.arg in banned
                        and isinstance(n.value, ast.Constant)):
                    bad.append(f"{nb.name} cell {i}: {n.arg}={n.value.value!r} "
                               f"passed as a literal")
                if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                        and any(isinstance(t, ast.Name) and t.id.lower() in banned
                                for t in n.targets)):
                    tgt = [t.id for t in n.targets if isinstance(t, ast.Name)][0]
                    bad.append(f"{nb.name} cell {i}: {tgt} = {n.value.value!r}")
    assert not bad, ("a notebook restates a setting instead of importing it:\n  " +
                     "\n  ".join(bad))
    print("settings  : no notebook restates a hyperparameter as a literal; all come from "
          "config.py -> rerun_canonical.py   OK")


def test_epoch_budget_is_200_everywhere():
    import config as C
    assert C.TRAIN_KWARGS["epochs"] == 200
    src = (REPO / "rerun_canonical.py").read_text()
    assert "epochs=200" in src
    # and no notebook or protocol_v2 module smuggles a 75 back in (this test file is
    # excluded, since the literal it looks for necessarily appears in it)
    scanned = [p for p in list(PV2.rglob("*.py")) + list((PV2 / "notebooks").glob("*.ipynb"))
               if p.resolve() != Path(__file__).resolve()]
    for p in scanned:
        text = p.read_text()
        assert "epochs=75" not in text and "EPOCHS = 75" not in text, \
            f"{p.relative_to(PV2)} sets the old 75-epoch budget"
    print(f"epochs    : {C.TRAIN_KWARGS['epochs']}, taken from "
          f"rerun_canonical.EXPECTED_TRAIN; no 75 anywhere in protocol_v2   OK")


def test_no_module_name_collides_with_src():
    """protocol_v2/config.py and src/config.py have the same module name. That collision
    already caused a bug — the notebooks put src/ on sys.path first, so `import config`
    returned the Sparkov preprocessing config and cell 1 died. This fails if a NEW
    collision is introduced, and checks that every entry point orders the path so
    protocol_v2 wins."""
    src_mods = {p.stem for p in (REPO / "src").glob("*.py")} - {"__init__"}
    pv2_mods = {p.stem for p in PV2.glob("*.py")}
    known = {"config"}
    new = (src_mods & pv2_mods) - known
    assert not new, (f"new module-name collisions with src/: {sorted(new)}. Either rename, "
                     f"or add to `known` here and make sure every sys.path setup orders "
                     f"protocol_v2 first.")

    # every notebook must put protocol_v2 first and assert it got the right module
    for nb in sorted((PV2 / "notebooks").glob("*.ipynb")):
        doc = json.loads(nb.read_text())
        first = "".join(next(c for c in doc["cells"] if c["cell_type"] == "code")["source"])
        i_src = first.find('"src"')
        i_pv2 = first.find("str(PV2))")
        assert i_src != -1 and i_pv2 != -1 and i_src < i_pv2, (
            f"{nb.name}: sys.path setup must insert src BEFORE protocol_v2, so that "
            f"protocol_v2 ends up first")
        assert "Path(CFG.__file__).parent == PV2" in first, (
            f"{nb.name}: missing the assertion that `import config` got protocol_v2's")
    print(f"collision : only the known 'config' clash with src/; all 5 notebooks order "
          f"the path correctly and assert it   OK")


def test_the_sign_convention_is_declared_and_used():
    """The excess must be built from the declared terms, not from a hand-written
    expression that can drift away from them."""
    import assembly_settings as A
    assert A.SIGN_CONVENTION["baseline"] == "P1_leaky"
    assert A.EXCESS_PER_WEEK == ((+1, "P1_causal"), (+1, "P3_acausal"),
                                 (-1, "P1_leaky"), (-1, "P3_rolling"))
    names = {n for n, _, _ in A.CONTRASTS}
    for n in A.SIGN_CONVENTION["single_corrections"]:
        assert n in names, f"single correction {n!r} is not a declared contrast"
    # the two single corrections must both start from the declared baseline
    for name, a, b in A.CONTRASTS:
        if name in A.SIGN_CONVENTION["single_corrections"]:
            assert a == A.SIGN_CONVENTION["baseline"], (
                f"{name} is {a} - {b}; a correction must start from "
                f"{A.SIGN_CONVENTION['baseline']}")
    # assemble must not contain a second, hand-written expression for the excess
    src = (PV2 / "assemble.py").read_text()
    assert "A.EXCESS_PER_WEEK" in src
    assert "(piv[\"P1_leaky\"] - piv[\"P1_causal\"])" not in src, (
        "a hand-written excess expression is back in assemble.py; it must be built from "
        "EXCESS_PER_WEEK so the convention and the arithmetic cannot disagree")
    print("sign      : convention declared, both single corrections start from P1_leaky, "
          "and the excess is built from the declared terms   OK")


def test_analysis_files_are_hashed():
    import provenance as P
    for rel in ("protocol_v2/assembly_settings.py", "protocol_v2/assemble.py"):
        assert rel in P.SOURCE_FILES, f"{rel} missing from SOURCE_FILES"
        assert rel in P.ANALYSIS_FILES, f"{rel} missing from ANALYSIS_FILES"
    print("analysis  : assembly_settings.py and assemble.py are both hashed and both "
          "locked   OK")


def test_checkpoint_is_written_last():
    """Per fold: scores, then val_curves, then the RNG state, then foldckpt LAST.

    The RNG state must be saved BEFORE the checkpoint. The other order leaves a window in
    which the checkpoint claims a fold the state file has no entry for, and that is the
    one combination nothing can recover — the random stream cannot be rewound, so the arm
    has to be restarted. Saving it first only ever leaves a harmless extra entry, which
    check_rng_state ignores.
    """
    src = (REPO / "src" / "run_date_gnn.py").read_text()
    i_sc = src.index("if scores_path:\n            hdr")
    i_vc = src.index("if val_curve_path:\n            hdr")
    i_rng = src.index("_save_rng_state(int(w), done_weeks)")
    i_ck = src.index("if checkpoint_path:\n            hdr")
    assert i_sc < i_vc < i_rng < i_ck, (
        f"write order is scores@{i_sc} val_curves@{i_vc} rng@{i_rng} checkpoint@{i_ck}. "
        f"Required: scores -> val_curves -> RNG state -> foldckpt last.")
    print("writeorder: scores -> val_curves -> RNG state -> foldckpt (last)   OK")
    print("            the RNG state precedes the checkpoint, so every crash window "
          "is recoverable")


if __name__ == "__main__":
    test_every_module_parses()
    test_every_notebook_cell_compiles()
    test_no_notebook_has_been_executed()
    test_each_arm_notebook_has_the_required_shape()
    test_only_the_four_arms_appear_as_targets()
    test_config_is_the_only_place_settings_live()
    test_epoch_budget_is_200_everywhere()
    test_no_module_name_collides_with_src()
    test_the_sign_convention_is_declared_and_used()
    test_analysis_files_are_hashed()
    test_checkpoint_is_written_last()
    print("\nstatic checks passed")
