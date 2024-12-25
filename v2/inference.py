import argparse

import cv2
import torch
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    # For image
    parser.add_argument('--image', type=str, default='image.png')
    # For model
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--mlp_dim', type=int, default=128)
    # For inference
    parser.add_argument('--ckpt', type=str, default='model.pth')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args


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
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.max(probs, dim=-1)
        result = ''
        for index in indices:
            result += chr(index + ord('a'))
    return result


def main(args):
    # Load model
    model = ViTCaptcha(
        image_size=28,
        patch_size=args.patch_size,
        num_classes=26,
        channels=1,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
    )
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(args.device)
    model.eval()
    # Read image
    image = read_image(args.image)
    # Segement
    images = segement(image)
    # Predict
    predict(images, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
