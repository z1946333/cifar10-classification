"""Evaluate a locally produced checkpoint on the official test set."""
import argparse
from datetime import datetime
from pathlib import Path

import torch

from datasets.cifar10 import CLASSES, test_loader
from models.resnet18 import build_model
from scripts.train import ROOT
from tools.plot_history import plot_confusion
from tools.runtime import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if saved['config'].get('smoke'):
        parser.error('Smoke checkpoints are not eligible for a formal test result. Use a full-training checkpoint.')
    model = build_model().to(device)
    model.load_state_dict(saved['model'])
    result = evaluate(model, test_loader(args.data_dir, 64, 0, device.type == 'cuda'), device)
    result.update(split='official_test', smoke=False, classes=CLASSES, checkpoint=str(args.checkpoint.resolve()),
                  best_epoch=saved['epoch'], accuracy_unit='fraction_0_to_1')
    output = ROOT / 'evaluation' / ('reevaluate_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    write_json(output / 'metrics.json', result)
    plot_confusion(result['confusion_matrix'], CLASSES, output / 'confusion_matrix.png')
    print('Test accuracy: {:.2%}; output: {}'.format(result['accuracy'], output))


if __name__ == '__main__':
    main()
