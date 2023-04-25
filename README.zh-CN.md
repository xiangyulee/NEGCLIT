|![overview](imgs/logo.jpg)|
|:--:|

# 网元与网图跨层协同推理与训练

[English](./README.md) | [简体中文](./README.zh-CN.md)


## 背景



|![overview](imgs/NEG.png)|
|:--:|
|图 1: 智能网元与网图关系
## 开始

NEGCLIT 可以部署在单机或者集群上，选择你需要的部署方法。
### 快速开始（单机部署）

<details open>
<summary>安装</summary>

克隆仓库并下载 [requirements.txt](./requirements.txt) 在
[**Python>=3.7.0**](https://www.python.org/) 环境下, 需要
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/xiangyulee/NEGCLIT  # clone
cd NEGCLIT
pip install -r requirements.txt  # install
```

</details>
<details open>
<summary>运行方案1</summary>

```bash
bash launch_ex1.sh
```

</details>

<details open>
<summary>运行方案2</summary>

```bash
bash launch_ex2.sh
```

</details>



### 集群部署
## 文档
### NEGCLIT 设计

- [工作流](./doc/workflow/README.zh_CN.md)
- [组件](./doc/component/README.zh_CN.md)
### 开发者资源

- [开发者向导](./doc/develop//README.zh_CN.md)
- [API文档](./doc/api/README.zh_CN.md)

### 测试结果
## 致谢

- [Fedlab](https://github.com/SMILELab-FL/FedLab)
- [Prune](https://github.com/Eric-mingjie/network-slimming)
- [OCL](https://github.com/RaptorMai/online-continual-learning)

## 开源协议

[Apache License 2.0](LICENSE)