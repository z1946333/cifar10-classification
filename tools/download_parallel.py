"""Optional parallel download from the official server, with full MD5 verification."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity
from tools.download_data import ensure_data, write_verified_stream


def main():
    root = Path(__file__).resolve().parents[1] / 'data'
    root.mkdir(parents=True, exist_ok=True)
    if check_integrity(str(root / CIFAR10.filename), CIFAR10.tgz_md5):
        ensure_data(root)
        print('Official archive already verified; no download needed.', flush=True)
        return
    with requests.Session() as session:
        response = session.head(CIFAR10.url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        url = response.url
        size = int(response.headers['Content-Length'])
    chunk_size = 8 * 1024 * 1024
    ranges = [(start, min(start + chunk_size - 1, size - 1)) for start in range(0, size, chunk_size)]
    temp = root / '.download_chunks'
    temp.mkdir(exist_ok=True)
    def fetch(index):
        start, end = ranges[index]
        target = Path(temp) / str(index)
        if target.exists() and target.stat().st_size == end - start + 1:
            return index
        offset = target.stat().st_size if target.exists() else 0
        if offset > end - start + 1:
            offset = 0
        request_start = start + offset
        with requests.Session() as session:
            with session.get(url, headers={'Range': 'bytes={}-{}'.format(request_start, end)}, stream=True, timeout=(30, 90)) as reply:
                reply.raise_for_status()
                expected = 'bytes {}-{}/{}'.format(request_start, end, size)
                if reply.status_code != 206 or reply.headers.get('Content-Range') != expected:
                    raise RuntimeError('Server did not honor range: ' + expected)
                with target.open('ab' if offset else 'wb') as output:
                    for chunk in reply.iter_content(256 * 1024):
                        output.write(chunk)
        if target.stat().st_size != end - start + 1:
            raise ValueError('Incomplete download range')
        return index

    def fetch_retry(index):
        for attempt in range(5):
            try:
                return fetch(index)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                print('Part {} retry {}: {}'.format(index, attempt + 1, type(exc).__name__), flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch_retry, i) for i in range(len(ranges))]
        failures = []
        for done, job in enumerate(as_completed(jobs), 1):
            try:
                print('Downloaded range {}/{} (part {})'.format(done, len(ranges), job.result()), flush=True)
            except Exception as exc:
                failures.append(exc)
                print('Range failed: {}'.format(exc), flush=True)
        if failures:
            raise RuntimeError('Some ranges failed. Run again to reuse completed ranges.') from failures[0]

    def chunks():
        for index in range(len(ranges)):
            with (Path(temp) / str(index)).open('rb') as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    yield block
    write_verified_stream(chunks(), root / CIFAR10.filename, CIFAR10.tgz_md5)
    ensure_data(root)


if __name__ == '__main__':
    main()
