## Reproducibility

This repository contains the source code used to build the graph datasets, train the MaGMA model, run inference, perform ablation testing, and evaluate regional forecast skill. It does not include the raw planetary datasets, generated graph objects, model checkpoints, or result outputs.

The raw datasets are publicly available through the DOIs cited in the manuscript. To reproduce the workflow, download the datasets locally, update the file paths in the scripts, and run the scripts in the following order:

1. `Graph Builder.py`
2. `Graph Loader.py`
3. `Graph Trainer.py`
4. `Test_multi year inference.py`
5. `Test_inference ablation.py`
6. `Test_my30-my34.py`
7. `Test_regional forecast skill.py`
