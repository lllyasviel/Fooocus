# Easy SDXL + Local Markup Run Guide

This build keeps Fooocus SDXL concepts but adds a simpler local control layer.

## Run Fooocus on Windows

Double-click:

```txt
RUN_FOOOCUS_LOCAL.bat
```

Or, from PowerShell after activating `.venv`:

```powershell
python launch.py --disable-analytics --listen
```

Fooocus opens at:

```txt
http://localhost:7865
```

## Run the Easy SDXL Control Center

Double-click:

```txt
RUN_MARKUP_ASSISTANT.bat
```

It opens at:

```txt
http://127.0.0.1:7871
```

## What the Easy SDXL Control Center does

It does not hide SDXL. It explains SDXL choices in human language:

- Generate New Image: use the main prompt box.
- Edit This Image: use Inpaint or Outpaint and draw a mask.
- Use Image as Reference: use the Image Prompt tab.
- Improve / Enhance: use the Enhance tab.

It recommends:

- Fooocus tab
- positive prompt
- inpaint prompt
- detection / mask prompt
- negative prompt
- steps
- CFG
- denoise guidance

## Correct editing workflow

1. Start Fooocus.
2. Start Easy SDXL Control Center.
3. Choose `Edit This Image`.
4. Choose target area such as `Glasses`, `Shirt / Clothes`, or `Face`.
5. Type the plain-English change.
6. Copy the generated inpaint prompt into Fooocus Inpaint.
7. Upload the image in `Inpaint or Outpaint`.
8. Paint the area to change.
9. Generate.

## Important

Do not reinstall requirements after CUDA Torch is working. If Torch is replaced by CPU-only Torch, reinstall:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Confirm:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

```txt
2.1.0+cu121
True
```

This is local-only. No RunPod or remote GPU changes are included.
