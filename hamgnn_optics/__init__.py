"""HamGNN optical postprocessing with explicit matrix and unit conventions."""

from .optics import (OpticalResult, RealSpaceMatrices, calculate_optical_properties,
                     write_optical_results)

__version__ = "0.1.0"
__all__ = [
    "RealSpaceMatrices", "OpticalResult", "calculate_optical_properties", "write_optical_results",
]

