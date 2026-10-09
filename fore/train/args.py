import argparse
from dataclasses import dataclass


@dataclass
class TrackerArgs:
    report_to: str = "none"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--report-to",
            type=str,
            default="none",
            choices=["none"],
            help="The integration to report results and logs to.",
        )
