import time
import os

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from model import ViTCaptcha


def read_image(path):
    image = cv2.imread(path)
    # Gray scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize
    image = cv2.resize(image, dsize=(0, 0), fx=28, fy=28)
    # Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Threshold
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return image


def segement(image):
    # Find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # Get result images
    images = []
    i = 0
    while i < len(contours):
        if i < len(contours) - 1:
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            x2, y2, w2, h2 = cv2.boundingRect(contours[i + 1])
            # x similar
            if abs(x1 - x2) < 56 or abs(x1 + w1 - x2 - w2) < 56:
                x, y = min(x1, x2), min(y1, y2)
                w, h = max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y
                image_i = image[y:y + h, x:x + w]
                # Resize
                image_i = cv2.resize(image_i, (28, 28))
                images.append(image_i)
                i += 2
                continue
        x, y, w, h = cv2.boundingRect(contours[i])
        image_i = image[y:y + h, x:x + w]
        # Resize
        image_i = cv2.resize(image_i, (28, 28))
        images.append(image_i)
        i += 1
    return images


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

    return captcha_correct / captcha_total, char_correct / char_total, captcha_correct, captcha_total, char_correct, char_total


def main():
    # Load model
    model = ViTCaptcha(
        image_size=28,
        patch_size=7,
        num_classes=26,
        channels=1,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        kv_heads=2
    )
    num_params = sum(p.numel() for p in model.parameters())
    ckpt = torch.load('../models/v2.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    # Prepare captcha files
    captcha_files = os.listdir('../labelled')
    all_time, preprocess_time, predict_time = 0, 0, 0
    captcha_preds, captcha_gts = [], []
    num_chars = 0
    # Inference
    for captcha_file in tqdm(captcha_files):
        preprocess_begin_time = time.time()
        # Read image
        image = read_image(f'../labelled/{captcha_file}')
        # Segement
        images = segement(image)
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
    captcha_acc, char_acc, captcha_correct, captcha_total, char_correct, char_total = metric(captcha_preds, captcha_gts)
    print(f'The number of parameters: {num_params / 1024:.2f}K')
    print(f'The accuracy of char: {100 * char_acc:.2f}% ({char_correct}/{char_total})')
    print(f'The accuracy of captcha: {100 * captcha_acc:.2f}% ({captcha_correct}/{captcha_total})')
    print(f'The fps of char is {num_chars / predict_time:.2f} ({num_chars} chars in {predict_time:.2f}s)')
    print(f'The fps of captcha is {len(captcha_files) / all_time:.2f} ({len(captcha_files)} captchas in {all_time:.2f}s)')
    print(f'The fps of preprocess is {len(captcha_files) / preprocess_time:.2f} ({len(captcha_files)} captchas in {preprocess_time:.2f}s)')


if __name__ == '__main__':
    main()
