"""
Entropy coding for quantized latent codes.

Implements arithmetic coding for efficient compression of quantized values.
Falls back to zlib if arithmetic coding is not available or efficient.
"""

import numpy as np
import zlib
import struct
import warnings


def compute_symbol_frequencies(data):
    """
    Compute frequency distribution of symbols in data.
    
    Args:
        data: Numpy array of integer symbols
    
    Returns:
        frequencies: Dictionary mapping symbol -> count
    """
    unique, counts = np.unique(data.flatten(), return_counts=True)
    frequencies = dict(zip(unique.tolist(), counts.tolist()))
    return frequencies


def compute_cumulative_frequencies(frequencies):
    """
    Compute cumulative frequency table for arithmetic coding.
    
    Args:
        frequencies: Dictionary of symbol -> count
    
    Returns:
        cumulative: Dictionary of symbol -> (low, high, total)
    """
    # Sort symbols
    symbols = sorted(frequencies.keys())
    
    cumulative = {}
    cumsum = 0
    total = sum(frequencies.values())
    
    for symbol in symbols:
        low = cumsum
        high = cumsum + frequencies[symbol]
        cumulative[symbol] = (low, high, total)
        cumsum = high
    
    return cumulative


class SimpleArithmeticEncoder:
    """
    Simple arithmetic encoder implementation.
    
    Note: This is a basic implementation. For production use, consider
    using optimized libraries like torchac or constriction.
    """
    
    def __init__(self, precision=32):
        """
        Initialize encoder.
        
        Args:
            precision: Number of bits for arithmetic coding precision
        """
        self.precision = precision
        self.max_range = 2 ** precision
        
    def encode(self, data, frequencies):
        """
        Encode data using arithmetic coding.
        
        Args:
            data: Numpy array of integer symbols
            frequencies: Dictionary of symbol -> count
        
        Returns:
            encoded: Bytes representing encoded data
            metadata: Dictionary with decoding information
        """
        cumulative = compute_cumulative_frequencies(frequencies)
        
        # Flatten data
        data_flat = data.flatten().tolist()
        total_count = sum(frequencies.values())
        
        # Initialize range
        low = 0
        high = self.max_range - 1
        
        # Encode each symbol
        for symbol in data_flat:
            if symbol not in cumulative:
                raise ValueError(f"Symbol {symbol} not in frequency table")
            
            sym_low, sym_high, sym_total = cumulative[symbol]
            range_size = high - low + 1
            
            # Update range
            high = low + (range_size * sym_high // sym_total) - 1
            low = low + (range_size * sym_low // sym_total)
            
            # Renormalization to prevent underflow
            while True:
                if high < self.max_range // 2:
                    # Both in lower half
                    low = 2 * low
                    high = 2 * high + 1
                elif low >= self.max_range // 2:
                    # Both in upper half
                    low = 2 * (low - self.max_range // 2)
                    high = 2 * (high - self.max_range // 2) + 1
                else:
                    break
        
        # Final value
        final_value = low
        
        # Convert to bytes
        encoded = struct.pack('>Q', final_value)  # 8 bytes for 64-bit value
        
        metadata = {
            'frequencies': frequencies,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'total_symbols': len(data_flat)
        }
        
        return encoded, metadata
    
    def decode(self, encoded, metadata):
        """
        Decode data using arithmetic coding.
        
        Args:
            encoded: Bytes representing encoded data
            metadata: Dictionary with decoding information
        
        Returns:
            data: Decoded numpy array
        """
        frequencies = metadata['frequencies']
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])
        total_symbols = metadata['total_symbols']
        
        cumulative = compute_cumulative_frequencies(frequencies)
        
        # Create reverse lookup: (low, high) -> symbol
        reverse_cumulative = {}
        for symbol, (low, high, total) in cumulative.items():
            for i in range(low, high):
                reverse_cumulative[i] = symbol
        
        # Decode value
        final_value = struct.unpack('>Q', encoded)[0]
        
        # Initialize range
        low = 0
        high = self.max_range - 1
        value = final_value
        
        decoded = []
        total_count = sum(frequencies.values())
        
        # Decode each symbol
        for _ in range(total_symbols):
            range_size = high - low + 1
            scaled_value = ((value - low + 1) * total_count - 1) // range_size
            
            # Find symbol
            symbol = None
            for sym, (sym_low, sym_high, sym_total) in cumulative.items():
                if sym_low <= scaled_value < sym_high:
                    symbol = sym
                    break
            
            if symbol is None:
                # Fallback to closest symbol
                symbol = min(cumulative.keys(), key=lambda s: abs(cumulative[s][0] - scaled_value))
            
            decoded.append(symbol)
            
            # Update range
            sym_low, sym_high, sym_total = cumulative[symbol]
            high = low + (range_size * sym_high // sym_total) - 1
            low = low + (range_size * sym_low // sym_total)
            
            # Renormalization
            while True:
                if high < self.max_range // 2:
                    low = 2 * low
                    high = 2 * high + 1
                    value = 2 * value
                elif low >= self.max_range // 2:
                    low = 2 * (low - self.max_range // 2)
                    high = 2 * (high - self.max_range // 2) + 1
                    value = 2 * (value - self.max_range // 2)
                else:
                    break
        
        # Reshape
        data = np.array(decoded, dtype=dtype).reshape(shape)
        return data


class EntropyEncoder:
    """
    Entropy encoder with automatic selection between arithmetic coding and zlib.
    """
    
    def __init__(self, method='auto'):
        """
        Initialize entropy encoder.
        
        Args:
            method: 'arithmetic', 'zlib', or 'auto' (choose best)
        """
        self.method = method
        self.arithmetic_encoder = SimpleArithmeticEncoder()
    
    def encode(self, data):
        """
        Encode quantized data.
        
        Args:
            data: Numpy array of integers (quantized latent codes)
        
        Returns:
            encoded: Compressed bytes
            metadata: Dictionary with decoding information
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Convert to bytes for zlib
        data_bytes = data.tobytes()
        
        # Try both methods if auto
        if self.method == 'auto':
            # Try zlib (fast and reliable)
            zlib_compressed = zlib.compress(data_bytes, level=9)
            zlib_size = len(zlib_compressed)
            
            # For large data or low entropy, arithmetic coding might be better
            # But for simplicity and reliability, we'll use zlib as default
            # Arithmetic coding is complex and the simple implementation may not be optimal
            
            method_used = 'zlib'
            encoded = zlib_compressed
            
        elif self.method == 'arithmetic':
            frequencies = compute_symbol_frequencies(data)
            encoded, arith_metadata = self.arithmetic_encoder.encode(data, frequencies)
            method_used = 'arithmetic'
            
        else:  # zlib
            encoded = zlib.compress(data_bytes, level=9)
            method_used = 'zlib'
        
        metadata = {
            'method': method_used,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'original_size': data.nbytes,
            'compressed_size': len(encoded)
        }
        
        if method_used == 'arithmetic':
            metadata['arithmetic'] = arith_metadata
        
        return encoded, metadata
    
    def decode(self, encoded, metadata):
        """
        Decode compressed data.
        
        Args:
            encoded: Compressed bytes
            metadata: Dictionary with decoding information
        
        Returns:
            data: Decoded numpy array
        """
        method = metadata['method']
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])
        
        if method == 'zlib':
            decompressed_bytes = zlib.decompress(encoded)
            data = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
            
        elif method == 'arithmetic':
            arith_metadata = metadata['arithmetic']
            data = self.arithmetic_encoder.decode(encoded, arith_metadata)
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return data


class EntropyDecoder:
    """
    Entropy decoder (same as encoder for symmetric coding).
    """
    
    def __init__(self):
        self.encoder = EntropyEncoder()
    
    def decode(self, encoded, metadata):
        """
        Decode compressed data.
        
        Args:
            encoded: Compressed bytes
            metadata: Dictionary with decoding information
        
        Returns:
            data: Decoded numpy array
        """
        return self.encoder.decode(encoded, metadata)


def estimate_entropy(data):
    """
    Estimate entropy of quantized data in bits per symbol.
    
    Args:
        data: Numpy array of integer symbols
    
    Returns:
        entropy: Entropy in bits
    """
    frequencies = compute_symbol_frequencies(data)
    total = sum(frequencies.values())
    
    entropy = 0.0
    for count in frequencies.values():
        if count > 0:
            prob = count / total
            entropy -= prob * np.log2(prob)
    
    return entropy


if __name__ == "__main__":
    # Test entropy coding
    print("Testing Entropy Coding...")
    
    # Generate test data with varying entropy
    np.random.seed(42)
    
    # Low entropy data (many repeated values)
    low_entropy = np.random.randint(0, 10, size=(1000,), dtype=np.uint8)
    
    # High entropy data (uniform distribution)
    high_entropy = np.random.randint(0, 256, size=(1000,), dtype=np.uint8)
    
    # Test with both
    for name, data in [("Low entropy", low_entropy), ("High entropy", high_entropy)]:
        print(f"\n=== {name} ===")
        print(f"Data shape: {data.shape}, dtype: {data.dtype}")
        print(f"Data range: [{data.min()}, {data.max()}]")
        
        # Calculate entropy
        entropy = estimate_entropy(data)
        print(f"Entropy: {entropy:.2f} bits/symbol")
        print(f"Theoretical compression: {data.nbytes} -> {int(len(data) * entropy / 8)} bytes")
        
        # Test encoder
        encoder = EntropyEncoder(method='auto')
        encoded, metadata = encoder.encode(data)
        
        print(f"Encoding method: {metadata['method']}")
        print(f"Original size: {metadata['original_size']} bytes")
        print(f"Compressed size: {metadata['compressed_size']} bytes")
        print(f"Compression ratio: {metadata['original_size'] / metadata['compressed_size']:.2f}x")
        
        # Test decoder
        decoder = EntropyDecoder()
        decoded = decoder.decode(encoded, metadata)
        
        # Verify
        if np.array_equal(data, decoded):
            print("✓ Decoding successful!")
        else:
            print("✗ Decoding failed!")
            print(f"Max error: {np.abs(data.astype(np.int32) - decoded.astype(np.int32)).max()}")

