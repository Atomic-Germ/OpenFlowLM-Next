# oflm-add cannot find the system registry on a real FLM install

## What happens

On this machine, with the released engine installed at `C:\Program Files\flm`
and `model_list.json` sitting right beside it, `oflm-add` refuses to run:

    Could not locate the system model_list.json (looked next to `oflm` and in
    /opt,/usr,/usr/local share/oflm). Pass --system-list.

Two separate faults, and the second is what makes the first fatal.

**The binary is called `flm`, not `oflm`.** `find_system_model_list` and
`find_system_xclbin_root` both call `shutil.which("oflm")` and nothing else, so
a standard install is invisible to them. The xclbin one returns None silently,
which turns into "[WARN] No xclbin source" rather than an error.

**`--system-list` is parsed and thrown away.** `main` calls
`find_system_model_list()` with no argument, so the flag the error message
tells the user to pass does nothing. There is no way out of the failure from
the command line.

## Requirement

New: `OPEN-ADD-SYSTEM-REGISTRY`. Nothing today specifies where `oflm-add` finds
the official registry, which is why neither fault was caught.

**Test category:** unit, in `utilities/oflm-add/tests/`.

## The change

- `find_system_model_list(explicit=None)` returns `explicit` when given, after
  checking it is a file, and `main` passes `args.system_list`.
- Both finders look for `flm` as well as `oflm`, `oflm` first so a checkout
  build still wins where both are on PATH.
- The refusal message names the paths actually tried.

No behaviour changes for a machine where `oflm` is on PATH.

## Not in scope

The registry's `details.quantization_level` being wrong for `gemma4-it:12b`
(see `.claude/plans/q4-0-container-sweep.md`). Its content is upstream's.
