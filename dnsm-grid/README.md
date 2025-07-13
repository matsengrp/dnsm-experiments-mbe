In this directory we have some infrastructure for trying out different combinations of things for DNSM models.
First, run Snakemake with the same base configurations (models, datasets, etc)
in `../dnsm-train/`, then run `snakemake -s [Snakefile]` in this directory, choosing `[Snakefile]` from
the list below.

* `optimizer_Snakefile`: for testing optimizers
* `train_hparam_Snakefile`: for testing training hyperparameters
* `multihit_Snakefile`: for testing with and without multihit correction
