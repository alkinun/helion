# Helion Examples

These scripts are intentionally small and focused. Run them from the
repository root after installing the package with `pip install -e .`.

Most examples require CUDA because Tritium launches Triton kernels.

## API Tours

- `modules_quickstart.py`: Helion modules and basic training utilities.
- `tritium_ops.py`: direct use of low-level Tritium kernels.

## Training And Inference

- `train_reference.py`: compact synthetic language-model training loop covering
  optimizers, schedulers, accumulation, AMP, EMA, clipping, and checkpoints.
- `inference_cached.py`: tiny character model plus KV-cache generation.
- `train_shakespeare.py`: practical tiny-shakespeare training and sampling.
- `generate_from_checkpoint.py`: load a `train_shakespeare.py` checkpoint and
  generate from a prompt.
- `evaluate_checkpoint.py`: load a `train_shakespeare.py` checkpoint and report
  held-out validation loss/perplexity.

Example train/save/test workflow:

```bash
python examples/train_shakespeare.py --steps 1000 --checkpoint-out /tmp/helion_shakespeare.pt
python examples/evaluate_checkpoint.py --checkpoint /tmp/helion_shakespeare.pt
python examples/generate_from_checkpoint.py --checkpoint /tmp/helion_shakespeare.pt --prompt "ROMEO:\n"
```

`train_shakespeare.py` trains on the first `1 - --val-fraction` portion of the
dataset and validates on the held-out tail. It logs validation loss/perplexity
every `--eval-interval` steps and on the final step. Add
`--best-checkpoint-out best.pt` to save the best validation checkpoint.

## Measurement

- `train_compare.py`: full training-step timing for Helion vs PyTorch eager
  and optional `torch.compile`.
