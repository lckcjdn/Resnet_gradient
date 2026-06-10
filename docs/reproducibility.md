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

## CIFAR-10 Real-data Rerun Commands

Download and verify CIFAR-10:

```bash
conda run -n resnet-gradient-path-study python -c "from torchvision import datasets, transforms; datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor()); datasets.CIFAR10(root='data', train=False, download=True, transform=transforms.ToTensor())"
```

Round 1:

```bash
conda run -n resnet-gradient-path-study python -m harness.run_suite --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2
```

Round 2:

```bash
conda run -n resnet-gradient-path-study python -m harness.run_suite --suite identity --dataset cifar10 --download --output-tag cifar10 --epochs 3 --train-size 2048 --val-size 512 --batch-size 128 --learning-rate 0.05 --device auto --torch-threads 2
```

Round 3:

```bash
conda run -n resnet-gradient-path-study python -m harness.lesion_study --dataset cifar10 --download --output-tag cifar10 --val-size 512 --batch-size 128 --device auto --torch-threads 2 --drop-ratios 0,0.1,0.3,0.5,0.7 --random-seeds 0,1,2
```

## Smoke Test Command

Run inside the project conda environment:

```bash
conda activate resnet-gradient-path-study
D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2
```

The smoke test uses a small subset and may fall back to `torchvision.datasets.FakeData` when CIFAR-10 is unavailable.

## Smoke Test Command

Run inside the project conda environment:

```bash
conda activate resnet-gradient-path-study
D:\研究生课程\python课程\ResNet\resnet-gradient-path-study\harness\run_suite.py --suite smoke --dataset cifar10 --download --output-tag cifar10 --epochs 2 --train-size 512 --val-size 256 --batch-size 64 --learning-rate 0.05 --device auto --torch-threads 2
```

The smoke test uses a small subset and may fall back to `torchvision.datasets.FakeData` when CIFAR-10 is unavailable.
