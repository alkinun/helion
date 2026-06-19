"""Run every committed benchmark module and print the results.

python -m benchmarks.run_all
TRITIUM_BENCH_REPEATS=7 python -m benchmarks.run_all
"""

from __future__ import annotations

import importlib
import pkgutil

import benchmarks


def main() -> None:
    for module_info in pkgutil.iter_modules(benchmarks.__path__):
        name = module_info.name
        if not name.startswith("bench_"):
            continue
        print(f"\n=== {name} ===")
        module = importlib.import_module(f"benchmarks.{name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            run = getattr(obj, "run", None)
            if callable(run):
                try:
                    run(print_data=True, show_plots=False)
                except TypeError:
                    run(print_data=True)


if __name__ == "__main__":
    main()
