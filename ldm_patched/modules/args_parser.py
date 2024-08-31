import argparse
import enum
import ldm_patched.modules.options

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0", help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--tls-keyfile", type=str, help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function")
parser.add_argument("--tls-keyfile-password", type=str, help="Password for the TLS (SSL) key file. Requires --tls-keyfile to function")
parser.add_argument("--tls-certfile", type=str, help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function")
parser.add_argument("--tls-verify", type=bool, default=True, help="If False, skips certificate validation which allows self-signed certificates to be used. Requires --tls-keyfile to function")
parser.add_argument("--hf-mirror", type=str, default=None, help="Use HuggingFace mirror URL")

parser.add_argument("--output-path", type=str, default=None, help="Set the output directory.")
parser.add_argument("--temp-path", type=str, default=None, help="Set the temp directory.")
parser.add_argument("--in-browser", action="store_true", help="Automatically launch in the default browser.")
parser.add_argument("--disable-in-browser", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID", help="Set the id of the cuda device this instance will use.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Run the UNET in bf16. This should only be used for testing stuff.")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Store unet weights in fp16.")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store text encoder weights in fp8 (e4m3fn variant).")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store text encoder weights in fp8 (e5m2 variant).")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")


parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")

parser.add_argument("--disable-ipex-optimize", action="store_true", help="Disables ipex.optimize when loading models with Intel GPUs.")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, help="Default preview method for sampler nodes.", action=EnumAction)

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the new pytorch 2.0 cross attention function.")

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force enable attention upcasting, please report if it fixes black images.")
upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable all upcasting of attention. Should be unnecessary except for debugging.")


vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--normalvram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--lowvram", action="store_true", help="Split the unet in parts to use less vram.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")


parser.add_argument("--disable-smart-memory", action="store_true", help="Force agressively offload to regular ram instead of keeping models in vram when it can.")
parser.add_argument("--deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--windows-standalone-build", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")

parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")

parser.add_argument("--verbose", action="store_true", help="Enables more debug prints.")


if ldm_patched.modules.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

if args.windows_standalone_build:
    args.in_browser = True

if args.disable_in_browser:
    args.in_browser = False

import logging
logging_level = logging.WARNING
if args.verbose:
    logging_level = logging.DEBUG

logging.basicConfig(format="%(message)s", level=logging_level)
