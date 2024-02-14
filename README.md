<div align="center">    
 
# ICAL: Implicit Character-Aided Learning for Enhanced Handwritten Mathematical Expression Recognition 

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
│       │   └── 
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
We have prepared the CROHME dataset in this GitHub repository, under the `data/crohme/` folder. 
The HME100K dataset is larger and requires downloading. We have provided a [download link](https://disk.pku.edu.cn/link/AAF68D1921C04943A685785F90F70A41EF). After downloading, please extract it to the `data/hme100k/` folder.

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
Next, navigate to ICAL folder and run `train.py`. It may take **32~33** hours on **4** NVIDIA 2080Ti gpus using ddp.
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

