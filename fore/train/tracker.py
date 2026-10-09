# tracker.py

import abc
from typing import Any, Dict, Optional

import torch.distributed as dist


class Tracker(abc.ABC):
    """
    Abstract Base Class for experiment trackers.

    Each tracker implementation should handle its own initialization, logging,
    and cleanup. It should also provide a class method to validate
    command-line arguments before initialization.
    """

    def __init__(self, args, output_dir: str):
        self.args = args
        self.output_dir = output_dir
        self.rank = dist.get_rank()
        self.is_initialized = False

    @classmethod
    @abc.abstractmethod
    def validate_args(cls, parser, args) -> None:
        """
        Validate necessary arguments for this tracker.
        This method is called during argument parsing.
        It should raise an error if required arguments are missing.
        """

    @abc.abstractmethod
    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to the tracker.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the tracker and clean up resources.
        """


class NoOpTracker(Tracker):
    """A tracker that does nothing, for when no tracking is desired."""

    @classmethod
    def validate_args(cls, parser, args):
        pass  # No arguments to validate

    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        self.is_initialized = True  # Considered initialized to do nothing

    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None):
        pass  # Do nothing

    def close(self):
        pass  # Do nothing


# --- Tracker Factory ---
TRACKER_REGISTRY = {
    "none": NoOpTracker,
}


def get_tracker_class(report_to: str) -> Optional[Tracker]:
    """Returns the tracker class based on the name."""
    return TRACKER_REGISTRY.get(report_to)


def create_tracker(args, output_dir: str) -> Tracker:
    """Factory function to create an experiment tracker instance."""
    tracker_class = get_tracker_class(args.report_to)
    if not tracker_class:
        raise ValueError(f"Unsupported report_to type: {args.report_to}")
    return tracker_class(args, output_dir)
