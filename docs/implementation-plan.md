# CIFAR-10 implementation plan

User requested automatic file creation and execution after reviewing the CIFAR-10 workflow.
Implement inline in a new independent directory, using the existing Python environment.

## Design
- PyTorch 2.5.0 / torchvision 0.20.0; CIFAR-adapted ResNet18, no pretrained weights.
- Official CIFAR-10, seeded 45000/5000 train/validation split; official 10000 test examples used only after selecting the best validation checkpoint.
- SGD, cosine schedule, 30 epochs, batch 64, seed 42, Windows workers=0, CUDA AMP.
- Separate smoke run with reduced real data and validation evaluation, clearly not a benchmark.
- Unique experiment/checkpoint/evaluation directories, atomic status and checkpoint writes; no overwriting previous results.
- Beginner launchers and Chinese documentation, no automatic GitHub publishing.

## Tasks
1. Write and run regression tests for reproducible disjoint splits, CIFAR model output/backpropagation, exact sample-weighted evaluation, empty input handling and run-directory collision protection.
2. Implement datasets/cifar10.py, models/resnet18.py and tools/runtime.py to satisfy tests.
3. Implement train/evaluate/plot entry points, configurations, launchers and documentation.
4. Run complete unit suite, download verified official data, run real CPU smoke training and checkpoint evaluation. CUDA execution waits because an unrelated rice training occupies the GPU.
5. Start full training in a hidden background process; inspect logs/status and report its actual state. Full results are generated automatically on completion.

## Verification
- python -m unittest discover -s tests -v
- python -m scripts.train --smoke
- Inspect metrics count, checkpoint, history, curves and status from the smoke run.
- Full run must report 45000 train, 5000 validation, 10000 test; a smoke result must never be presented as a final test result.
