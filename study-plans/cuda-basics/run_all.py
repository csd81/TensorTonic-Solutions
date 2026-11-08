"""Run the self-test of every Numba port.

Usage:
    python run_all.py                 # every problem
    python run_all.py relu softmax    # selected problems
"""

import importlib.util
import pathlib
import sys
import traceback
import warnings

sys.dont_write_bytecode = True

try:
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent
ONLY = set(sys.argv[1:])


def iter_modules():
    for d in sorted(ROOT.iterdir()):
        path = d / f"{d.name}.py"
        if d.is_dir() and path.exists() and (not ONLY or d.name in ONLY):
            yield d.name, path


failures = []
for name, path in iter_modules():
    spec = importlib.util.spec_from_file_location("numba_" + name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.test()
    except Exception:
        failures.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("all tests passed")
