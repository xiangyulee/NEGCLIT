|![overview](imgs/logo.jpg)|
|:--:|

# NEGCLIT Overview

[English](./README.md) | [简体中文](./README.zh-CN.md)

## Background

NEGCLIT (Network Elements and Graph Cross Layer Inference and Training) is a federated learning framework to enable users to train an early exit network online,such that both client and server can make prediction. 

|![overview](imgs/NEG.png)|
|:--:|
|figure 1: relationship between intelligent network element and graph|
## Getting Started

NEGCLIT can be deployed on a single host or on multiple nodes. Choose the deployment approach which matches your environment.
### Quick Start（Standalone deployment）

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](./requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/xiangyulee/NEGCLIT  # clone
cd NEGCLIT
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Run</summary>

```bash
bash launch.sh
```

</details>


### Cluster deployment
## Documentation
### NEGCLIT Design 

- [Workflow](./doc/workflow/README.md)
- [Components](./doc/component/README.md)
### Developer Resources

- [Developer Guide for NEGCLIT](./doc/develop//README.md)
- [EGCLIT API references](./doc/api/README.md)
## test results
## Acknowledgments

- [Fedlab](https://github.com/SMILELab-FL/FedLab)
- [Prune](https://github.com/Eric-mingjie/network-slimming)
- [OCL](https://github.com/RaptorMai/online-continual-learning)

## License

[Apache License 2.0](LICENSE)