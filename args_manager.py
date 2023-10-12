from cbh.options import enable_args_parsing
enable_args_parsing(False)
import cbh.cli_args as cbh_cli


cbh_cli.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

cbh_cli.args = cbh_cli.parser.parse_args()
cbh_cli.args.disable_cuda_malloc = True
cbh_cli.args.auto_launch = True

if getattr(cbh_cli.args, 'port', 8188) == 8188:
    cbh_cli.args.port = None

args = cbh_cli.args
