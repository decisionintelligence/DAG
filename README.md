# <img src="docs/figures/icon.png" alt="Image description" style="width:27px;height:27px;"> DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables

 [![ICML](https://img.shields.io/badge/ICML'26-DAG-orange)](https://arxiv.org/pdf/2509.14933) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)

This code is the official PyTorch implementation of our ICML'26 paper, [DAG](https://arxiv.org/pdf/2509.14933): A Dual Correlation Network for Time Series Forecasting with Exogenous Variables.

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!


🚩 News (2025.10) We have open-sourced the [covariate time series forecasting leaderboard](https://decisionintelligence.github.io/OpenTS/leaderboards/#covariate_forecasting).


## Introduction
DAG, which utilizes Dual correlAtion network along both the temporal and channel dimensions for time series forecasting with exoGenous variables, especially leveraging future exogenous variable information. 

The framework introduces a Temporal Correlation Module which includes a temporal correlation discovery module to model how historical exogenous variables affect future exogenous variables, followed by a correlation injection module to incorporate these relationships for forecasting future endogenous variables. 

Additionally, a Channel Correlation Module is introduced. It features a channel correlation discovery module to model the impact of historical exogenous variables on historical endogenous variables and injects these relationships for improved forecasting of future endogenous variables based on future exogenous variables.

<div align="center">
<img alt="Logo" src="docs/figures/main.png" width="100%"/>
</div>

## Quickstart

> [!IMPORTANT]
> this project is fully tested under python 3.8, it is recommended that you set the Python version to 3.8.
1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

2. Data preparation

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1K2AvogpOpSz1PiQ53dPchzGv_PqlCWAK/view?usp=sharing). Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model

- To see the model structure of DAG,  [click here](./ts_benchmark/baselines/dag/models/dag_model.py).
- We provide all the experiment scripts for DAG and other baselines under the folder `./scripts/covariate_forecasting`.  For example you can reproduce all the experiment results as the following script:

```shell
sh ./scripts/w_future/DAG.sh
```



## Results
We utilize the Time Series Forecasting Benchmark (TFB) code repository as a unified evaluation framework, providing access to **all baseline codes, scripts, and results**. Following the settings in TFB, we do not apply the **"Drop Last"** trick to ensure a fair comparison.

<div align="center">
<img alt="Logo" src="docs/figures/result1.png" width="100%"/>
</div>

## Citation

If you find this repo useful, please cite our paper.

```

@inproceedings{qiu2026dag,
  title        =   {DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables},
  author       =   {Qiu, Xiangfei and Zhu, Yuhan and Li, Zhengyu and Wu, Xingjian and Yang, Bin and Hu, Jilin},
  booktitle    =   {ICML},
  year         =   {2026}
}
```



## Contact

If you have any questions or suggestions, feel free to contact:

- [Xiangfei Qiu](https://qiu69.github.io/) (xfqiu@stu.ecnu.edu.cn)

Or describe it in Issues.
