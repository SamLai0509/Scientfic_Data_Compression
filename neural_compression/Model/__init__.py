"""
Model module for NeurLZ compression.

Exports all model classes used in NeurLZ compression.
"""

import sys
import os

# Add current directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# =============================================================================
# Import all model classes from their respective files
# =============================================================================

# Simple spatial residual predictor
from simple_model import TinyResidualPredictor

# Frequency-based models with different input configurations
from frequency_1_input_model_a import TinyFrequencyResidualPredictor_1_input
from frequency_4_inputs_model_b import TinyFrequencyResidualPredictor_4_inputs
from frequency_7_inputs_model_c import TinyFrequencyResidualPredictor_7_inputs

# Energy-based spatial frequency model
from spatial_energy_model import TinyFrequencyResidualPredictorWithEnergy

# =============================================================================
# Aliases for backward compatibility
# =============================================================================

# TinyFrequencyResidualPredictor is an alias for the 1-input version
TinyFrequencyResidualPredictor = TinyFrequencyResidualPredictor_1_input

# 3_mag_3_phase refers to 7 inputs (1 spatial + 3 mag bands + 3 phase bands)
TinyFrequencyResidualPredictor_3_mag_3_phase = TinyFrequencyResidualPredictor_7_inputs

# =============================================================================
# Export all classes
# =============================================================================
__all__ = [
    # Primary classes
    'TinyResidualPredictor',
    'TinyFrequencyResidualPredictor_1_input',
    'TinyFrequencyResidualPredictor_4_inputs',
    'TinyFrequencyResidualPredictor_7_inputs',
    'TinyFrequencyResidualPredictorWithEnergy',
    # Aliases
    'TinyFrequencyResidualPredictor',
    'TinyFrequencyResidualPredictor_3_mag_3_phase',
]
