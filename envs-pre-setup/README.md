# MARL Environments
---

This directory contains source code customized for PettingZoo Parallel api
- SMAC with support latest PZ parallel api (Tested)
- SMAC (v2) with support latest PZ parallel api (Still working on this since the current version can not load v1-like parallel env)
- GFootBall with support latest PZ parallel api (Tested)

The underlying engine is the same as original implementations.

---

## Set up
I assume that the conda env named `pymarl` exists.

```bash
conda activate pymarl
sh setup.sh
```