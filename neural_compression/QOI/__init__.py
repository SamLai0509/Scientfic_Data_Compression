# lightweight registry (optional)
from .aggregator import QoIAggregator
from .spectral import SpectralPowerQoI
from .divergence import DivergenceQoI

__all__ = ["QoIAggregator", "SpectralPowerQoI", "DivergenceQoI"]