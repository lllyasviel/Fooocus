Quick troubleshooting checklist for Fooocus

- If UI not reachable: check process and port
  - PowerShell: Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'python_embeded' } | Select ProcessId,CommandLine
  - netstat -ano | Select-String ":7865"
- Tail logs: Get-Content .\fooocus_windows_run_from_download_embedded.log -Wait -Tail 200
- If DLL import errors (numpy/onnxruntime/cv2): prefer conda-installed binaries (use conda-forge/pytorch channels)
- For OpenMP conflicts: set KMP_DUPLICATE_LIB_OK=TRUE (temporary)
- If GPU issues: update NVIDIA driver and ensure CUDA/PyTorch match
- If environment broken: use official Windows release (run.bat) or Docker
- If models fail: delete corrupted files in models/checkpoints and restart to re-download

Keep full logs (fooocus_*.log) when filing issues.