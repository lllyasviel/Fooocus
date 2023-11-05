from fcbh.options import enable_args_parsing
enable_args_parsing(False)
import fcbh.cli_args as fcbh_cli


fcbh_cli.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
fcbh_cli.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")

fcbh_cli.parser.add_argument("--language", type=str, default='default',
                             help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# For example, https://github.com/lllyasviel/Fooocus/issues/849
fcbh_cli.parser.add_argument("--enable-smart-memory", action="store_true",
                             help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

fcbh_cli.parser.set_defaults(
    disable_cuda_malloc=True,
    auto_launch=True,
    port=None
)

fcbh_cli.args = fcbh_cli.parser.parse_args()

# Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724
fcbh_cli.args.disable_smart_memory = not fcbh_cli.args.enable_smart_memory

args = fcbh_cli.args
