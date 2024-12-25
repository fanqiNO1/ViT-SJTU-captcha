import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from model import ViTCaptcha
from dataset import AlphaBeta


def parse_args():
    parser = argparse.ArgumentParser()
    # For dataset
    parser.add_argument('--path', type=str, default='dataset')
    # For dataloader
    parser.add_argument('--batch_size', type=int, default=32)
    # For model
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--mlp_dim', type=int, default=128)
    parser.add_argument('--kv_heads', type=int, default=2)
    # For training
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    # For logging and saving
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='models')
    args = parser.parse_args()
    return args


def train(model, dataloader, optimizer, criterion, epoch, device, log_interval):
    model.train()
    for batch_idx, data in enumerate(dataloader):
        image, label = data
        image, label = image.to(device), label.to(device)
        # Forward
        output = model(image)
        loss = criterion(output, label)
        # Accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(label.view_as(pred)).sum().item()
        accuracy = correct / len(image)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            logging.info(
                f'Train Epoch: {epoch} [{(batch_idx + 1) * len(image)}/{len(dataloader.dataset)} '
                f'({100. * (batch_idx + 1) / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}\t'
                f'Accuracy: {accuracy:.6f}'
            )

def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            image, label = data
            image, label = image.to(device), label.to(device)
            # Forward
            output = model(image)
            test_loss += criterion(output, label).item()
            # Accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    # Create save path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # Create dataset
    logging.info('Creating dataset...')
    train_dataset = AlphaBeta(args.path, is_training=True)
    test_dataset = AlphaBeta(args.path, is_training=False)
    # Create dataloader
    logging.info('Creating dataloader...')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Create model
    logging.info('Creating model...')
    model = ViTCaptcha(
        image_size=28,
        patch_size=args.patch_size,
        num_classes=26,
        channels=1,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        kv_heads=args.kv_heads,
    )
    model = model.to(args.device)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params = num_params / 1e6
    logging.info(f'Number of parameters: {num_params:.2f}M')
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # Training
    for epoch in range(1, args.epochs + 1):
        logging.info(f'Epoch {epoch}')
        train(model, train_dataloader, optimizer, criterion, epoch, args.device, args.log_interval)
        test(model, test_dataloader, criterion, args.device)
        scheduler.step()
        # Save model
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'model_{epoch}.pth'))


if __name__ == '__main__':
    args = parse_args()
    main(args)