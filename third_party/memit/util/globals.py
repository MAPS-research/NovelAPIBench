# Modified: upstream reads these paths from a `globals.yml` in the working directory.
# MEMIT only uses STATS_DIR (the covariance cache), which novelapibench.adaptation.memit sets
# explicitly on memit.memit_main before editing.
from pathlib import Path

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
STATS_DIR = Path("data/stats")
HPARAMS_DIR = Path("hparams")
KV_DIR = Path("kvs")

REMOTE_ROOT_URL = "https://memit.baulab.info"
