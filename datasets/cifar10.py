"""Official CIFAR-10 with a fixed train/validation split."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def split_indices(total=50000, val_size=5000, seed=42):
    if not 0 < val_size < total:
        raise ValueError('Validation size must be between 1 and total - 1.')
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def transform(training=False):
    items = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if training else []
    return transforms.Compose(items + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_loaders(root, batch_size=64, seed=42, workers=0, smoke=False, pin_memory=False):
    from tools.download_data import ensure_data
    ensure_data(root)
    # Separate dataset objects ensure validation never receives random augmentation.
    augmented = datasets.CIFAR10(str(root), train=True, download=False, transform=transform(True))
    plain = datasets.CIFAR10(str(root), train=True, download=False, transform=transform(False))
    train_ids, val_ids = split_indices(len(augmented), 5000, seed)
    if smoke:
        train_ids, val_ids = train_ids[:128], val_ids[:64]
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
    train = DataLoader(Subset(augmented, train_ids), shuffle=True,
                       generator=torch.Generator().manual_seed(seed), **options)
    val = DataLoader(Subset(plain, val_ids), shuffle=False, **options)
    return train, val, {'train_indices': train_ids, 'validation_indices': val_ids, 'seed': seed}


def test_loader(root, batch_size=64, workers=0, pin_memory=False):
    from tools.download_data import ensure_data
    ensure_data(root)
    dataset = datasets.CIFAR10(str(root), train=False, download=False, transform=transform(False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=pin_memory)
