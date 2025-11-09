"""
Pure Neural Network Data Compressor

A standalone neural network-based compressor for scientific 3D data
with GPU support and strict error bound enforcement.

No traditional compressor (SZ3, etc.) required - pure neural compression.
"""

from .neural_autoencoder_3d import NeuralAutoencoder3D, NeuralEncoder3D, NeuralDecoder3D
from .neural_compressor import NeuralCompressor
from .quantization import UniformQuantizer, AdaptiveQuantizer
from .entropy_coding import EntropyEncoder, EntropyDecoder

__version__ = "1.0.0"

__all__ = [
    'NeuralAutoencoder3D',
    'NeuralEncoder3D',
    'NeuralDecoder3D',
    'NeuralCompressor',
    'UniformQuantizer',
    'AdaptiveQuantizer',
    'EntropyEncoder',
    'EntropyDecoder',
]

