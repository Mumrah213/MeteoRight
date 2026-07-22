"""Entry point: parse arguments and dispatch to a command module.

Command implementations are imported lazily inside `main` so that
`--help` and argument errors do not pay for pandas, matplotlib and the
analysis stack.
"""

from __future__ import annotations

import logging
import sys

from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch event-verification commands.
    if args.command == "advanced":
        from .advanced import cmd_advanced_compare, cmd_advanced_events, cmd_advanced_skill

        adv_commands = {
            "events": cmd_advanced_events,
            "skill": cmd_advanced_skill,
            "compare": cmd_advanced_compare,
        }
        handler = adv_commands.get(args.advanced_command)
        if handler:
            return handler(args)
        parser._meteoright_subparsers.choices["advanced"].print_help()
        return 1

    # Dispatch dataset commands.
    if args.command == "prepare-dataset":
        from datasets.cli import _handle_prepare_dataset

        return _handle_prepare_dataset(args)

    if args.command == "dataset-info":
        from datasets.cli import _handle_dataset_info

        return _handle_dataset_info(args)

    # Grid point finder
    if args.command == "grid-points":
        from .grid import cmd_grid_points

        return cmd_grid_points(args)

    # Standard commands
    if args.command == "download":
        from .download import cmd_download

        return cmd_download(args)
    if args.command == "verify":
        from .verify import cmd_verify

        return cmd_verify(args)
    if args.command == "metrics":
        from .metrics import cmd_metrics

        return cmd_metrics(args)
    if args.command == "analyze":
        from .analyze import cmd_analyze

        return cmd_analyze(args)
    if args.command == "pipeline":
        from .pipeline import cmd_pipeline

        return cmd_pipeline(args)
    if args.command == "blend":
        from .blend import cmd_blend

        return cmd_blend(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
