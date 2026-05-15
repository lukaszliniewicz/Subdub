from .app import run_app
from .cli_args import build_parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_app(args, parser)


if __name__ == "__main__":
    main()
