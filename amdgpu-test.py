import torch
import subprocess

print("=" * 40)
print("PyTorch ROCm Environment Test")
print("=" * 40)

# Run PyTorch environment diagnostics
print("\nRunning PyTorch Environment Diagnostics...\n")
subprocess.run(["python3", "-m", "torch.utils.collect_env"])
