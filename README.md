<div align="center">    
 
# ICAL: Implicit Character-Aided Learning for Enhanced Handwritten Mathematical Expression Recognition 
[![arXiv](https://img.shields.io/badge/arXiv-2405.09032-b31b1b.svg)](https://arxiv.org/abs/2405.09032)

</div>

## Project structure
```bash
├── config/         # config for ICAL hyperparameter
├── data/
│   └── crohme      # CROHME Dataset
│   └── HME100k      # HME100k Dataset which needs to be downloaded according to the instructions below.
├── eval/             # evaluation scripts
├── ical               # model definition folder
├── lightning_logs      # training logs
│   └── version_0      # ckpt for CROHME dataset
│       ├── checkpoints
│       │   └── epoch=125-step=47375-val_ExpRate=0.6101.ckpt
│       ├── config.yaml
│       └── hparams.yaml
│   └── version_1      # ckpt for HME100k dataset
│       ├── checkpoints
│       │   └── epoch=55-step=175503-val_ExpRate=0.6924.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd ICAL
# install project   
conda create -y -n ICAL python=3.7
conda activate ICAL
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
 ```
## Dataset Preparation
We have prepared the CROHME dataset and HME100K dataset in [download link](https://disk.pku.edu.cn/link/AAF10CCC4D539543F68847A9010C607139). After downloading, please extract it to the `data/` folder.

## Training on CROHME Dataset
Next, navigate to ICAL folder and run `train.py`. It may take **8~9** hours on **4** NVIDIA 2080Ti gpus using ddp.
```bash
# train ICAL model using 4 gpus and ddp on CROHME dataset
python -u train.py --config config/crohme.yaml
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
```

## Training on HME100k Dataset
It may take about **48** hours on **4** NVIDIA 2080Ti gpus using ddp on HME100k dataset.
```bash
# train ICAL model using 4 gpus and ddp on hme100k dataset
python -u train.py --config config/hme100k.yaml
```

## Evaluation
Trained ICAL weight checkpoints for CROHME and HME100K Datasets have been saved in `lightning_logs/version_0` and `lightning_logs/version_1`, respectively.

```bash
# For CROHME Dataset
bash eval/eval_crohme.sh 0

# For HME100K Dataset
bash eval/eval_hme100k.sh 1
```
## Experimental Results
Given the relative small size of the CROHME dataset, we conducted experiments with both the baseline CoMER and the proposed ICAL model using five different random seeds (7, 77, 777, 7777, 77777) under the same experimental conditions. The reported results are the averages and standard deviations of these five experiments.

| Method    | CROHME 2014 ExpRate↑ | CROHME 2014 ≤1↑ | CROHME 2014 ≤2↑ | CROHME 2016 ExpRate↑ | CROHME 2016 ≤1↑ | CROHME 2016 ≤2↑ | CROHME 2019 ExpRate↑ | CROHME 2019 ≤1↑ | CROHME 2019 ≤2↑ |
|-----------|----------------------|-----------------|-----------------|----------------------|-----------------|-----------------|----------------------|-----------------|-----------------|
| BTTR      | 53.96                | 66.02           | 70.28           | 52.31                | 63.90           | 68.61           | 52.96                | 65.97           | 69.14           |
| GCN*      | 60.00                | -               | -               | 58.94                | -               | -               | 61.63                | -               | -               |
| CoMER†    | 58.38±0.62           | 74.48±1.41      | 81.14±0.91      | 56.98±1.41           | 74.44±0.93      | 81.87±0.73      | 59.12±0.43           | 77.45±0.70      | 83.87±0.80      |
| **ICAL**  | **60.63±0.61**       | **75.99±0.77**  | **82.80±0.40**  | **58.79±0.73**       | **76.06±0.37**  | **83.38±0.16**  | **60.51±0.71**       | **78.00±0.66**  | **84.63±0.45**  |


| Method    | HME100K ExpRate↑ | HME100K ≤1↑    | HME100K ≤2↑    |
|-----------|------------------|----------------|----------------|
| DenseWAP  | 61.85            | 70.63          | 77.14          |
| DenseWAP-TD | 62.60          | 79.05          | 85.67          |
| ABM       | 65.93            | 81.16          | 87.86          |
| SAN       | 67.1             | -              | -              |
| CAN-DWAP  | 67.31            | 82.93          | 89.17          |
| CAN-ABM   | 68.09            | 83.22          | 89.91          |
| BTTR      | 64.1             | -              | -              |
| CoMER†    | 68.12            | 84.20          | 89.71          |
| **ICAL**  | **69.06±0.16**   | **85.16±0.13** | **90.61±0.09** |


### Citation   
```
@article{zhu2024ical,
      title={ICAL: Implicit Character-Aided Learning for Enhanced Handwritten Mathematical Expression Recognition}, 
      author={Jianhua Zhu and Liangcai Gao and Wenqi Zhao},
      year={2024},
      eprint={2405.09032},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      
}
```

## Reference
- [CoMER](https://github.com/Green-Wood/CoMER) | [arXiv](https://arxiv.org/abs/2207.04410)
- [BTTR](https://github.com/Green-Wood/BTTR) | [arXiv](https://arxiv.org/abs/2105.02412)
- [TreeDecoder](https://github.com/JianshuZhang/TreeDecoder)
- [CAN](https://github.com/LBH1024/CAN) | [arXiv](https://arxiv.org/abs/2207.11463)

