"""Edit this file to customize input and output locations.

To keep your modifications out of version control, you can run the following command:

git update-index --skip-worktree dnsmex/local.py
"""

import os

LOCAL_PATHS = {
    "DATA_DIR": "~/data",
    "MISC": "~/data/misc",
    "SABDAB": "~/sabdab",
    "FIGURES_DIR": "~/writing/talks/figures/bcr-mut-sel/",
    "DNSM_TRAINED_MODELS_DIR": "~/re/dnsm-experiments-mbe/dnsm-train/trained_models",
    "DDSM_TRAINED_MODELS_DIR": "~/re/dnsm-experiments-1/ddsm-train/trained_models",
    "DASM_TRAINED_MODELS_DIR": "~/re/dnsm-experiments-1/dasm-train/trained_models",
    "DNSM_TEST_OUTPUT_DIR": "~/re/dnsm-experiments-1/dnsm-train/_ignore/test_output",
    "DDSM_TEST_OUTPUT_DIR": "~/re/dnsm-experiments-1/ddsm-train/_ignore/test_output",
    "DASM_TEST_OUTPUT_DIR": "~/re/dnsm-experiments-1/dasm-train/_ignore/test_output",
    "DASM_GRID_DIR": "~/re/dnsm-experiments-1/dasm-grid/trained_models",
    "DNSM_GRID_DIR": "~/re/dnsm-experiments-1/dnsm-grid/trained_models",
    "DDSM_GRID_DIR": "~/re/dnsm-experiments-1/ddsm-grid/trained_models",
}


def localify(path):
    for key, value in LOCAL_PATHS.items():
        path = path.replace(key, value)
    path = path.replace("~", os.path.expanduser("~"))
    return path
