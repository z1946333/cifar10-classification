import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


class CoreTests(unittest.TestCase):
    def test_process_identity_rejects_reused_pid(self):
        import os
        from tools.launch import is_running, process_start_time
        birth = process_start_time(os.getpid())
        self.assertIsNotNone(birth)
        self.assertTrue(is_running(os.getpid(), birth))
        self.assertFalse(is_running(os.getpid(), birth + 1))

    def test_launcher_lock_recovers_after_owner_exits(self):
        import subprocess
        import sys
        from tools.launch import launcher_lock
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lock'
            command = [sys.executable, '-c',
                       'import os,sys; from pathlib import Path; from tools.launch import launcher_lock; '
                       'lock=launcher_lock(Path(sys.argv[1])); lock.__enter__(); os._exit(0)', str(path)]
            completed = subprocess.run(command, check=False, timeout=20)
            self.assertEqual(completed.returncode, 0)
            with launcher_lock(path):
                with self.assertRaises(BlockingIOError):
                    with launcher_lock(path):
                        pass

    def test_download_rejects_corrupt_bytes_and_preserves_existing_file(self):
        import hashlib
        from tools.download_data import write_verified_stream
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'archive.bin'
            target.write_bytes(b'existing')
            with self.assertRaises(ValueError):
                write_verified_stream([b'corrupt'], target, hashlib.md5(b'correct').hexdigest())
            self.assertEqual(target.read_bytes(), b'existing')
            self.assertEqual(list(Path(temp).glob('*.part')), [])
            write_verified_stream([b'cor', b'rect'], target, hashlib.md5(b'correct').hexdigest())
            self.assertEqual(target.read_bytes(), b'correct')

    def test_split_is_reproducible_disjoint_and_complete(self):
        from datasets.cifar10 import split_indices
        train, val = split_indices(50000, 5000, 42)
        self.assertEqual(len(train), 45000)
        self.assertEqual(len(val), 5000)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(50000)))
        self.assertEqual((train, val), split_indices(50000, 5000, 42))
        self.assertNotEqual(val, split_indices(50000, 5000, 43)[1])

    def test_invalid_split_is_rejected(self):
        from datasets.cifar10 import split_indices
        for size in (0, 10, 11):
            with self.assertRaises(ValueError):
                split_indices(10, size, 42)

    def test_cifar_model_supports_ten_classes_and_training(self):
        from models.resnet18 import build_model
        torch.set_num_threads(2)
        model = build_model()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 10))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 9])).backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertGreater(model.conv1.weight.grad.abs().sum().item(), 0)

    def test_evaluation_counts_samples_not_batches(self):
        from tools.runtime import evaluate
        logits = torch.full((3, 10), -3.0)
        logits[0, 0] = 3.0
        logits[1, 1] = 3.0
        logits[2, 1] = 3.0
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = evaluate(torch.nn.Identity(), loader, torch.device('cpu'))
        self.assertEqual(result['samples'], 3)
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['loss'], torch.nn.functional.cross_entropy(logits, labels).item(), places=5)
        self.assertEqual(result['confusion_matrix'][0][1], 1)
        self.assertEqual(sum(map(sum, result['confusion_matrix'])), 3)

    def test_empty_evaluation_is_rejected(self):
        from tools.runtime import evaluate
        loader = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)))
        with self.assertRaises(ValueError):
            evaluate(torch.nn.Identity(), loader, torch.device('cpu'))

    def test_previous_experiment_is_never_overwritten(self):
        from tools.runtime import create_run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = create_run(root, 'test_run')
            self.assertTrue(paths['experiment'].is_dir())
            with self.assertRaises(FileExistsError):
                create_run(root, 'test_run')
            for name in ('../escape', 'bad/name', ''):
                with self.assertRaises(ValueError):
                    create_run(root, name)


if __name__ == '__main__':
    unittest.main()
