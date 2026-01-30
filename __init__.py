"""
Neural-Enhanced SZ3 Compression
Based on NeurLZ approach: https://arxiv.org/abs/2409.05785
"""

from .residual_encoder_3d import ResidualEncoder3D
from .neurlz_sz3_compression import NeuralSZ3Compressor

__all__ = ['ResidualEncoder3D', 'NeuralSZ3Compressor']

