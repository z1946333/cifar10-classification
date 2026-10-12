import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def create_run(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', name):
        raise ValueError('Run name must contain only letters, digits, underscores and hyphens.')
    paths = {'experiment': root / 'experiments' / name,
             'checkpoint': root / 'checkpoints' / name,
             'evaluation': root / 'evaluation' / name}
    for path in paths.values():
        if path.exists():
            raise FileExistsError('Run already exists: ' + str(path))
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    return paths


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, count = 0.0, 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss_sum += criterion(output, labels).item()
        prediction = output.argmax(1)
        confusion += torch.bincount((labels * 10 + prediction).cpu(), minlength=100).reshape(10, 10)
        count += labels.numel()
    if count == 0:
        raise ValueError('Cannot evaluate an empty dataset.')
    return {'loss': loss_sum / count, 'accuracy': confusion.diag().sum().item() / count,
            'samples': count, 'confusion_matrix': confusion.tolist()}
