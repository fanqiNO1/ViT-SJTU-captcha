import os
import time

import cv2
import torch
from model import ViTCaptcha
from segment import segment
from torch import nn
from torchvision import transforms
from tqdm import tqdm


def predict(images, model):
    # ToTensor
    image = torch.stack([transforms.ToTensor()(image) for image in images])
    # Predict
    with torch.inference_mode():
        logits = model(image)
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.max(probs, dim=-1)
        result = ''
        for index in indices:
            result += chr(index + ord('a'))
    return result


def metric(preds, gts):
    assert len(preds) == len(gts)
    captcha_correct, captcha_total = 0, 0
    char_correct, char_total = 0, 0

    for (pred, gt) in zip(preds, gts):
        captcha_total += 1
        if pred == gt:
            captcha_correct += 1

        if len(pred) == len(gt):
            char_total += len(gt)
            for (p, g) in zip(pred, gt):
                if p == g:
                    char_correct += 1

    return (captcha_correct / captcha_total, char_correct / char_total,
            captcha_correct, captcha_total, char_correct, char_total)


def main():
    # Load model
    model = ViTCaptcha(image_size=28,
                       num_layers=2,
                       num_classes=26,
                       patch_size=7,
                       in_channels=1,
                       hidden_size=16,
                       num_attention_heads=4,
                       num_key_value_heads=1,
                       intermediate_size=32,
                       act_fn=nn.SiLU())
    num_params = sum(p.numel() for p in model.parameters())
    ckpt = torch.load('../../models/v3.pth',
                      map_location='cpu',
                      weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    # Prepare captcha files
    captcha_files = os.listdir('../../labelled')
    all_time, preprocess_time, predict_time = 0, 0, 0
    captcha_preds, captcha_gts = [], []
    num_chars = 0
    # Inference
    for captcha_file in tqdm(captcha_files):
        preprocess_begin_time = time.time()
        # Read image
        image = cv2.imread(f'../../labelled/{captcha_file}')
        # Segement
        images = segment(image)
        images = [cv2.resize(image, (28, 28)) for image in images]
        preprocess_end_time = time.time()
        # Predict
        predict_begin_time = time.time()
        captcha_pred = predict(images, model)
        predict_end_time = time.time()
        # Record
        captcha_preds.append(captcha_pred)
        captcha_gts.append(captcha_file.split('_')[0])
        all_time += predict_end_time - preprocess_begin_time
        preprocess_time += preprocess_end_time - preprocess_begin_time
        predict_time += predict_end_time - predict_begin_time
        num_chars += len(captcha_pred)
    # Metric
    (captcha_acc, char_acc, captcha_correct, captcha_total, char_correct,
     char_total) = metric(captcha_preds, captcha_gts)
    print(f'The number of parameters: {num_params / 1024:.2f}K')
    print(f'The accuracy of char: {100 * char_acc:.2f}% '
          f'({char_correct}/{char_total})')
    print(f'The accuracy of captcha: {100 * captcha_acc:.2f}% '
          f'({captcha_correct}/{captcha_total})')
    print(f'The fps of char is {num_chars / predict_time:.2f} '
          f'({num_chars} chars in {predict_time:.2f}s)')
    print(f'The fps of captcha is {len(captcha_files) / all_time:.2f} '
          f'({len(captcha_files)} captchas in {all_time:.2f}s)')
    print(
        f'The fps of preprocess is {len(captcha_files) / preprocess_time:.2f} '
        f'({len(captcha_files)} captchas in {preprocess_time:.2f}s)')


if __name__ == '__main__':
    main()
