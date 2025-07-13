# Project Context for Claude

## Repository Structure

This repository (`dnsm-experiments-1`) is for evaluating and training netam models on various datasets. It will be published alongside relevant papers.

## Related Repositories

- **netam** (located at `/Users/matsen/re/netam/`): The main repository containing the model implementations and core functionality
  - Import netam modules with: `from netam.common import ...`, `from netam.framework import ...`, etc.

## Environment Setup

To activate the correct Python environment for working with netam:

```bash
source ~/re/netam/.venv/bin/activate
```

## Common Imports

```python
from netam.common import *
from netam.framework import load_crepe
from netam.sequences import AA_STR_SORTED
```