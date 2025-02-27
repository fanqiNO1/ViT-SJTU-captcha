import torch
import torch.nn as nn
from dataset import AlphaBeta
from mmengine.evaluator import BaseMetric
from mmengine.hooks import CheckpointHook
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.optim import CosineAnnealingLR, LinearLR
from mmengine.runner import Runner
from model import ViTCaptcha


class Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = ViTCaptcha(image_size=28,
                                num_layers=2,
                                num_classes=26,
                                patch_size=7,
                                in_channels=1,
                                hidden_size=16,
                                num_attention_heads=4,
                                num_key_value_heads=1,
                                intermediate_size=32,
                                act_fn=nn.SiLU())
        self.loss = nn.CrossEntropyLoss()

    def log_params(self):
        params = sum(p.numel() for p in self.model.parameters())
        if params > 1e6:
            params = f'{params / 1e6:.2f}M'
        elif params > 1e3:
            params = f'{params / 1e3:.2f}K'
        print_log(f'Number of parameters: {params}', logger='current')

    def forward(self, x, y, mode):
        output = self.model(x)
        if mode == 'loss':
            return {'loss': self.loss(output, y)}
        elif mode == 'predict':
            return output, y


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


def main():
    model = Model()

    train_set = AlphaBeta(data_path='dataset', is_training=True)
    test_set = AlphaBeta(data_path='dataset', is_training=False)

    batch_size = 32
    train_loader = dict(batch_size=batch_size,
                        num_workers=0,
                        dataset=train_set,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        collate_fn=dict(type='default_collate'))
    test_loader = dict(batch_size=batch_size,
                       num_workers=0,
                       dataset=test_set,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       collate_fn=dict(type='default_collate'))

    max_epochs = 30
    runner = Runner(
        model=model,
        work_dir='./models',
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=1),
        val_cfg=dict(),
        optim_wrapper=dict(optimizer=dict(type=torch.optim.AdamW, lr=5e-3)),
        param_scheduler=[
            dict(type=LinearLR,
                 start_factor=1e-3,
                 by_epoch=True,
                 begin=0,
                 end=int(max_epochs * 0.2),
                 convert_to_iter_based=True),
            dict(type=CosineAnnealingLR,
                 eta_min=0.0,
                 by_epoch=True,
                 begin=int(max_epochs * 0.2),
                 end=max_epochs,
                 convert_to_iter_based=True)
        ],
        val_evaluator=dict(type=Accuracy),
        default_hooks=dict(checkpoint=dict(
            type=CheckpointHook, max_keep_ckpts=1, save_best='auto')),
        randomness=dict(seed=0x66ccff))
    model.log_params()
    runner.train()


if __name__ == '__main__':
    main()
