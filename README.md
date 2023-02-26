# VIT-SJTU-captcha

## Introduction

A captcha solver for SJTU based on ViT and some preprocessing tricks. The accuracy of character classification is 99.6%.

## Getting Started

```bash
pip install -r requirements.txt
```

## Usage

### Inference

```python
python inference.py --image image.png --model model.pth
```

### Training

```python
python main.py --path dataset
```

## Dataset

The dataset is from [here](https://github.com/PhotonQuantum/jaccount-captcha-solver/releases/tag/v2.0)

### Preprocessing

1. Segment the captcha into some images with one character.
2. Split the dataset into train and test sets.
