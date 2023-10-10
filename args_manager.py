from comfy.options import enable_args_parsing
enable_args_parsing(False)
import comfy.cli_args as comfy_cli


comfy_cli.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

comfy_cli.args = comfy_cli.parser.parse_args()
comfy_cli.args.disable_cuda_malloc = True
comfy_cli.args.auto_launch = True

if getattr(comfy_cli.args, 'port', 8188) == 8188:
    comfy_cli.args.port = None

args = comfy_cli.args
