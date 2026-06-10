# Reproducibility

## Conda Environment

Preferred setup:

```bash
conda env create -f environment.yml
conda activate resnet-gradient-path-study
python -m harness.sanity_check
```

Local setup used in this run:

```bash
conda create -y -n resnet-gradient-path-study --clone E:\python\envs\MeMOTR
conda run -n resnet-gradient-path-study python -m pip install pandas
conda run -n resnet-gradient-path-study python -m harness.sanity_check
```

The direct `conda create -n resnet-gradient-path-study python=3.10` command failed because the machine could not reach the default Anaconda package repository. Cloning a local conda environment provided an isolated project environment while avoiding network package resolution.

## Smoke Test Command

Run inside the project conda environment:

```bash
conda activate resnet-gradient-path-study
D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset fake --epochs 1 --train-size 96 --val-size 48 --batch-size 24 --learning-rate 0.01 --device cpu --torch-threads 2
```

The smoke test uses a small subset and may fall back to `torchvision.datasets.FakeData` when CIFAR-10 is unavailable.
