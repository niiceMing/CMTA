# CMTA
(NIPS23)[Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning](https://openreview.net/forum?id=WIrZh2XxLT) 

This repository is based on [MTRL](https://github.com/facebookresearch/mtrl). Please refer to the documentation of MTRL for any installation issues or guidance.
# Notice 
We use experiment.random_pos to choose Fixed environment or Mixed environment, and add corresponding code in src/mtenv, so it doesn't need to git clone submodule mtenv anymore.
# Getting Started 
We should install the local mtenv lib at first:
```bash
pip install -e src/mtenv/.
```
Then you can use the following instructions to run CMTA:
```bash
cd scripts
bash CMTA.sh $seed$
```
\$ seed \$ can be 1,10,42,...
