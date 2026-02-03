# Fooocus CLI (Command Line Interface)

This is a custom Command Line Interface for Fooocus, allowing you to generate images programmatically without launching the web browser. It supports advanced features like model selection, custom styles, LoRAs, and batch generation.

## 🚀 Quick Start

From the Fooocus directory, run:

```bash
python entry_with_update.py --prompt "a beautiful landscape" --output "landscape.png"
```

For Windows Standalone users (assuming you are in the install folder):
```powershell
.\python_embeded\python.exe Fooocus\entry_with_update.py --prompt "test"
```

## 🛠️ Usage & Arguments

### Basic Usage

```bash
python entry_with_update.py --prompt "your prompt" --aspect-ratio "1024x1024"
```

### Advanced Usage

```bash
python entry_with_update.py ^
  --prompt "cyberpunk detective, neon rain" ^
  --negative-prompt "bright, sunny" ^
  --steps 30 ^
  --cfg-scale 7.0 ^
  --style "Fooocus V2" --style "Fooocus Cyberpunk" ^
  --base-model "juggernautXL_v8Rundiffusion.safetensors" ^
  --lora "sd_xl_offset_example-lora_1.0.safetensors:0.5" ^
  --image-number 2 ^
  --output "detective.png"
```

### Argument Reference

| Category | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| **Core** | `--prompt` | **(Required)** The positive prompt text. | N/A |
| | `--negative-prompt` | valid negative prompt text. | "" |
| | `--output` | Output filename. Relative command run location. | N/A |
| | `--seed` | Seed number for reproducibility. `-1` is random. | -1 |
| | `--image-number` | Number of images to generate in a row. | 1 |
| **Performance** | `--performance` | Preset: `Speed`, `Quality`, `Extreme Speed`. | Speed |
| | `--steps` | Exact number of sampling steps (overrides performance). | N/A |
| | `--aspect-ratio` | Dimensions (e.g., `1152x896`, `1024x1024`). | 1152x896 |
| | `--cfg-scale` | Guidance scale (how strictly to follow prompt). | 7.0 |
| | `--sharpness` | Image sharpness filter strength. | 2.0 |
| | `--sampler` | Sampler method name. | dpmpp_2m_sde_gpu |
| | `--scheduler` | Scheduler name. | karras |
| **Models** | `--base-model` | Filename of the base checkpoint. | (Config Default) |
| | `--refiner-model` | Filename of the refiner checkpoint. | (Config Default) |
| | `--refiner-switch` | Step ratio to switch to refiner (0.0-1.0). | 0.5 |
| | `--lora` | Load LoRA: `filename:weight`. Use flag multiple times. | N/A |
| **Styles** | `--style` | Style name. Use flag multiple times. | (Fooocus Defaults) |

## 🤖 AI Agent Integration

If you want to teach an AI agent to use this tool, provide it with the following specification:

### Tool: `generate_image_fooocus`

**Description:** Generates images locally using Fooocus via CLI.
**Execution:** `d:\Fooocus_win64_2-5-0\fooocus-cli.bat`

**Parameters:**
*   `prompt`: String (Required)
*   `negative_prompt`: String
*   `output`: String (Filename)
*   `aspect_ratio`: String (e.g., "1024x1024")
*   `base_model`: String (Checkpoint filename)
*   `style`: List[String] (Style names)
*   `lora`: List[String] (Format "name:weight")
*   `steps`: Integer
*   `cfg_scale`: Float

**Notes:**
*   Output is saved relative to `d:\Fooocus_win64_2-5-0\Fooocus\` if a relative path is given in python, but the batch wrapper usually handles CWD. Absolute paths are recommended for the `--output` argument to ensure files are saved exactly where intended.
