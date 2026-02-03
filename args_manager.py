import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true',
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='default',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# For example, https://github.com/lllyasviel/Fooocus/issues/849
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                help="Disables enhance output sorting for final image gallery.")

args_parser.parser.add_argument("--enable-auto-describe-image", action='store_true',
                                help="Enables automatic description of uov and enhance image when prompt is empty", default=False)

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

# CLI Arguments
args_parser.parser.add_argument("--prompt", type=str, default=None, help="Positive prompt for CLI mode.")
args_parser.parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for CLI mode.")
args_parser.parser.add_argument("--seed", type=int, default=-1, help="Seed for CLI mode.")
args_parser.parser.add_argument("--output", type=str, default=None, help="Output filename for CLI mode.")
args_parser.parser.add_argument("--performance", type=str, default="Speed", choices=["Speed", "Quality", "Extreme Speed"], help="Performance mode for CLI mode.")
args_parser.parser.add_argument("--aspect-ratio", type=str, default="1152\u00d7896", help="Aspect ratio for CLI mode.")
args_parser.parser.add_argument("--base-model", type=str, default=None, help="Base model filename.")
args_parser.parser.add_argument("--refiner-model", type=str, default=None, help="Refiner model filename.")
args_parser.parser.add_argument("--refiner-switch", type=float, default=0.5, help="Refiner switch point (0.0-1.0).")
args_parser.parser.add_argument("--lora", action='append', help="LoRA definition in format 'filename:weight'. Can be specified multiple times.")
args_parser.parser.add_argument("--style", action='append', help="Style name. Can be specified multiple times.")
args_parser.parser.add_argument("--cfg-scale", type=float, default=7.0, help="CFG Scale (default 7.0).")
args_parser.parser.add_argument("--sampler", type=str, default="dpmpp_2m_sde_gpu", help="Sampler name.")
args_parser.parser.add_argument("--scheduler", type=str, default="karras", help="Scheduler name.")
args_parser.parser.add_argument("--sharpness", type=float, default=2.0, help="Image sharpness (default 2.0).")
args_parser.parser.add_argument("--image-number", type=int, default=1, help="Number of images to generate (default 1).")
args_parser.parser.add_argument("--steps", type=int, default=-1, help="Number of steps (overrides performance).")

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()

# (Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724)
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args
