#!/usr/bin/env python
"""
Thin wrapper around robomimic's training entry point.

Stock ``robomimic.scripts.train`` does not know about the flow_matching algorithm,
so importing ``robomimic_cfm`` first registers it, then we hand off to robomimic's
own ``train.py`` __main__ (argparse and all) unchanged. Use exactly like
robomimic's trainer:

    python scripts/train.py --config configs/benchmark/fm_square_seed1.json
    python scripts/train.py --config <cfg> --resume
"""
import runpy

import robomimic_cfm  # noqa: F401  (registers the "flow_matching" algo + config)

if __name__ == "__main__":
    # run robomimic's train.py as __main__ so its full CLI / arg parsing applies,
    # with flow_matching already registered in this interpreter
    runpy.run_module("robomimic.scripts.train", run_name="__main__", alter_sys=True)
