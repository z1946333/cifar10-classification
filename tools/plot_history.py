import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_history(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Training history is empty.')
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ('loss', 'accuracy')):
        for phase in ('train', 'val'):
            ax.plot(epochs, [float(row[phase + '_' + key]) for row in rows], label=phase)
        ax.set(xlabel='Epoch', ylabel=key.title())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(csv_path.parent.name)
    fig.tight_layout()
    fig.savefig(csv_path.with_name('curves.png'), dpi=160)
    plt.close(fig)


def plot_confusion(matrix, classes, output):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted class', ylabel='True class', title='Confusion matrix (counts)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center',
                    color='white' if matrix[row, col] > matrix.max() / 2 else 'black', fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('history', type=Path)
    plot_history(parser.parse_args().history)
