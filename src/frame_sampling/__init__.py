"""Reusable video frame sampling strategies.

Independent of any downstream task (no imports from action_ontologies), so it
can be lifted into other projects wholesale.
"""

from .information_gain import SampledFrame, sample_by_information_gain

__all__ = ["SampledFrame", "sample_by_information_gain"]
