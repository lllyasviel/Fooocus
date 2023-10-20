from fcbh.options import enable_args_parsing
enable_args_parsing(False)
import fcbh.cli_args as fcbh_cli


fcbh_cli.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
fcbh_cli.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")

fcbh_cli.parser.add_argument("--language", type=str, default='default',
                             help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

fcbh_cli.args = fcbh_cli.parser.parse_args()
fcbh_cli.args.disable_cuda_malloc = True
fcbh_cli.args.auto_launch = True

if getattr(fcbh_cli.args, 'port', 8188) == 8188:
    fcbh_cli.args.port = None

args = fcbh_cli.args
