"""Train, select on validation, then evaluate the selected model once on test."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch import nn

from datasets.cifar10 import CLASSES, make_loaders, test_loader
from models.resnet18 import build_model
from tools.plot_history import plot_confusion, plot_history
from tools.runtime import create_run, evaluate, save_checkpoint, seed_everything, write_json

ROOT = Path(__file__).resolve().parents[1]


def gpu_free_mb():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    return int(result.stdout.strip().splitlines()[0])


def train_epoch(model, loader, optimizer, scaler, device, amp):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(images)
            loss = criterion(output, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; inspect data/configuration.')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * labels.size(0)
        correct += (output.argmax(1) == labels).sum().item()
        count += labels.size(0)
        if step % 100 == 0:
            print('  batch {}/{} loss={:.4f} accuracy={:.2%}'.format(step, len(loader), loss_sum / count, correct / count), flush=True)
    return loss_sum / count, correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/resnet18.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--smoke', action='store_true', help='128 train / 64 validation examples, NOT a test benchmark')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--run-name')
    parser.add_argument('--wait-for-gpu', action='store_true', help='Wait for GPU 0 to have sufficient free VRAM')
    parser.add_argument('--min-free-mb', type=int, default=1800)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.smoke:
        config.update(epochs=1, batch_size=16, workers=0)
    if config['epochs'] < 1 or config['batch_size'] < 1 or config['workers'] < 0:
        parser.error('epochs/batch_size must be positive and workers nonnegative')
    if config['learning_rate'] <= 0:
        parser.error('learning_rate must be positive')
    run_name = args.run_name or ('smoke_' if args.smoke else 'resnet18_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    paths = create_run(ROOT, run_name)
    status_path = paths['experiment'] / 'status.json'
    state = {'run_name': run_name, 'pid': os.getpid(), 'smoke': args.smoke, 'stage': 'starting',
             'epochs_total': config['epochs'], 'epoch': 0,
             'started_at': datetime.now().astimezone().isoformat()}

    def status(**updates):
        state.update(updates)
        state['updated_at'] = datetime.now().astimezone().isoformat()
        write_json(status_path, state)

    try:
        status()
        config.update(smoke=args.smoke, data_dir=str(args.data_dir.resolve()), model='cifar_resnet18',
                      pretrained=False, selection_metric='validation_accuracy', run_name=run_name)
        write_json(paths['experiment'] / 'config.json', config)
        print('RUN: ' + run_name, flush=True)
        print('STATUS: ' + str(status_path), flush=True)
        # Query nvidia-smi before creating a CUDA context, so queued runs consume no VRAM.
        if args.wait_for_gpu and args.device != 'cpu':
            status(stage='waiting_for_gpu', required_free_mb=args.min_free_mb)
            while True:
                free = gpu_free_mb()
                status(free_gpu_mb=free)
                if free >= args.min_free_mb:
                    break
                print('Waiting for GPU: {} MiB free; need {} MiB. Existing tasks are not stopped.'.format(free, args.min_free_mb), flush=True)
                time.sleep(30)
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this Python environment.')
        torch.set_num_threads(4)
        seed_everything(config['seed'])
        amp = bool(config['amp'] and device.type == 'cuda')
        environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                       'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                       'cuda_runtime': torch.version.cuda, 'device': str(device),
                       'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                       'amp': amp, 'deterministic_cudnn': True,
                       'note': 'Seeded split and training; bitwise equivalence across hardware/software is not guaranteed.'}
        write_json(paths['experiment'] / 'environment.json', environment)
        source_files = [p for folder in ('models', 'datasets', 'scripts', 'tools') for p in (ROOT / folder).glob('*.py')]
        write_json(paths['experiment'] / 'source_hashes.json', {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        status(stage='loading_data', device=str(device))
        train, val, split = make_loaders(args.data_dir, config['batch_size'], config['seed'],
                                         config['workers'], args.smoke, device.type == 'cuda')
        write_json(paths['experiment'] / 'split_indices.json', split)
        print('Device: {}; train: {}; validation: {}'.format(device, len(train.dataset), len(val.dataset)), flush=True)
        model = build_model().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        history_path = paths['experiment'] / 'history.csv'
        fields = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'learning_rate', 'seconds']
        best_accuracy, best_epoch = -1.0, 0
        training_start = time.perf_counter()
        with history_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            for epoch in range(1, config['epochs'] + 1):
                status(stage='training', current_epoch=epoch)
                start = time.perf_counter()
                learning_rate = optimizer.param_groups[0]['lr']
                train_loss, train_accuracy = train_epoch(model, train, optimizer, scaler, device, amp)
                validation = evaluate(model, val, device)
                scheduler.step()
                if validation['accuracy'] > best_accuracy:
                    best_accuracy, best_epoch = validation['accuracy'], epoch
                    save_checkpoint(paths['checkpoint'] / 'best.pt', {
                        'model': model.state_dict(), 'epoch': epoch, 'validation_accuracy': best_accuracy,
                        'config': config, 'classes': CLASSES})
                save_checkpoint(paths['checkpoint'] / 'last.pt', {
                    'model': model.state_dict(), 'epoch': epoch, 'validation_accuracy': validation['accuracy'],
                    'config': config, 'classes': CLASSES})
                row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                           val_loss=validation['loss'], val_accuracy=validation['accuracy'],
                           learning_rate=learning_rate, seconds=time.perf_counter() - start)
                writer.writerow(row)
                handle.flush()
                plot_history(history_path)
                status(epoch=epoch, train_accuracy=train_accuracy, validation_accuracy=validation['accuracy'],
                       best_validation_accuracy=best_accuracy, best_epoch=best_epoch, epoch_seconds=row['seconds'])
                print('Epoch {}/{} | train {:.2%} | validation {:.2%} | {:.1f}s'.format(
                    epoch, config['epochs'], train_accuracy, validation['accuracy'], row['seconds']), flush=True)
        training_seconds = time.perf_counter() - training_start
        status(stage='evaluating_best_checkpoint')
        checkpoint = torch.load(paths['checkpoint'] / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        final_loader = val if args.smoke else test_loader(args.data_dir, config['batch_size'], config['workers'], device.type == 'cuda')
        result = evaluate(model, final_loader, device)
        result.update(run_name=run_name, split='validation_smoke' if args.smoke else 'official_test',
                      smoke=args.smoke, best_epoch=best_epoch, best_validation_accuracy=best_accuracy,
                      training_seconds=training_seconds, classes=CLASSES, accuracy_unit='fraction_0_to_1')
        result['per_class_accuracy'] = {
            name: (result['confusion_matrix'][i][i] / sum(result['confusion_matrix'][i])
                   if sum(result['confusion_matrix'][i]) else None) for i, name in enumerate(CLASSES)}
        write_json(paths['evaluation'] / 'metrics.json', result)
        plot_confusion(result['confusion_matrix'], CLASSES, paths['evaluation'] / 'confusion_matrix.png')
        report = ('# ' + run_name + '\n\n'
                  + ('**SMOKE TEST ONLY: not a CIFAR-10 test-set result.**\n\n' if args.smoke else '')
                  + '- Model: CIFAR-adapted ResNet18, trained from scratch\n'
                  + '- Split: {} ({} samples)\n'.format(result['split'], result['samples'])
                  + '- Accuracy: {:.4%}\n'.format(result['accuracy'])
                  + '- Best validation epoch: {}\n'.format(best_epoch)
                  + '- Best validation accuracy: {:.4%}\n'.format(best_accuracy)
                  + '- Training time: {:.1f} seconds\n'.format(training_seconds)
                  + '- Device: {}\n'.format(environment['gpu'] or environment['device'])
                  + '- Seed: {}\n\n'.format(config['seed'])
                  + '![Training curves](../../experiments/{}/curves.png)\n\n'.format(run_name)
                  + '![Confusion matrix](confusion_matrix.png)\n')
        (paths['evaluation'] / 'report.md').write_text(report, encoding='utf-8')
        status(stage='completed', metrics_path=str(paths['evaluation'] / 'metrics.json'),
               final_accuracy=result['accuracy'], training_seconds=training_seconds)
        if not args.smoke:
            write_json(ROOT / 'evaluation/latest.json', {'run_name': run_name, 'metrics': str(paths['evaluation'] / 'metrics.json')})
        print('COMPLETED: {} accuracy={:.2%}; {}'.format(result['split'], result['accuracy'], paths['evaluation']), flush=True)
    except BaseException as exc:
        status(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        (paths['experiment'] / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
