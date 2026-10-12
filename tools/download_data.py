"""Download official CIFAR-10 with TLS verification and the torchvision MD5."""
import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import requests
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity


def write_verified_stream(chunks, destination, expected_md5):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=str(destination.parent))
    os.close(descriptor)
    temporary = Path(name)
    digest = hashlib.md5()
    received, reported = 0, 0
    try:
        with temporary.open('wb') as handle:
            for chunk in chunks:
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if received - reported >= 10 * 1024 * 1024:
                        print('Downloaded {:.1f} MiB'.format(received / 1024**2), flush=True)
                        reported = received
        if digest.hexdigest() != expected_md5:
            raise ValueError('CIFAR-10 archive checksum mismatch; refusing to extract.')
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_data(root):
    root = Path(root)
    try:
        CIFAR10(str(root), train=True, download=False)
        CIFAR10(str(root), train=False, download=False)
        return
    except RuntimeError:
        pass
    archive = root / CIFAR10.filename
    if not check_integrity(str(archive), CIFAR10.tgz_md5):
        print('Downloading official CIFAR-10: ' + CIFAR10.url, flush=True)
        # requests uses its CA bundle, avoiding a broken Windows certificate store.
        # Certificate validation remains enabled.
        with requests.get(CIFAR10.url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            write_verified_stream(response.iter_content(1024 * 1024), archive, CIFAR10.tgz_md5)
    # torchvision verifies the archive and the individual extracted batch files.
    train = CIFAR10(str(root), train=True, download=True)
    test = CIFAR10(str(root), train=False, download=False)
    print('Verified official CIFAR-10: {} train, {} test'.format(len(train), len(test)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    ensure_data(parser.parse_args().data_dir)
