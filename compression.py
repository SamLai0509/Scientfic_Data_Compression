import numpy as np
import sys
import os

sys.path.append('/Users/923714256/Data_compression/SZ3/tools/pysz')
from pysz import SZ

def compress_f32_file(input_file, output_file, shape=(512, 512, 512), eb_mode=0, eb_abs=1e-6, eb_rel=0, eb_pwr=0):
    """
    Compress a .f32 binary file using SZ3
    
    Parameters:
    - input_file: path to input .f32 file
    - output_file: path to output compressed file
    - shape: data dimensions (default: 512x512x512)
    - eb_mode: error bound mode (0:ABS, 1:REL, 2:ABS_AND_REL, 3:ABS_OR_REL, 4:PSNR, 5:NORM, 10:PW_REL)
    - eb_abs: absolute error bound
    - eb_rel: relative error bound  
    - eb_pwr: pointwise relative error bound
    """
    
    # Initialize SZ with the library path
    sz = SZ("/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so")
    
    # Read the .f32 binary file into numpy array
    data = np.fromfile(input_file, dtype=np.float32).reshape(shape)
    
    print(f"Processing: {os.path.basename(input_file)}")
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")
    print(f"Data range: [{np.min(data):.6e}, {np.max(data):.6e}]")
    
    # Compress the data
    data_cmpr, cmpr_ratio = sz.compress(data, eb_mode, eb_abs, eb_rel, eb_pwr)
    
    print(f"Compression ratio: {cmpr_ratio:.2f}")
    print(f"Original size: {data.nbytes / (1024**2):.2f} MB")
    print(f"Compressed size: {data_cmpr.nbytes / (1024**2):.2f} MB")
    print(f"Compressed data shape: {data_cmpr.shape}, dtype: {data_cmpr.dtype}")
    
    # Save compressed data
    data_cmpr.tofile(output_file)
    print(f"Saved to: {output_file}\n")
    
    return cmpr_ratio

def decompress_f32_file(input_file, output_file, shape=(512, 512, 512)):
    """
    Decompress a .f32 binary file using SZ3
    
    Parameters:
    - input_file: path to input compressed file
    - output_file: path to output decompressed file
    - shape: data dimensions (default: 512x512x512)
    """
    # Initialize SZ with the library path
    sz = SZ("/Users/923714256/Data_compression/SZ3/build/lib64/libSZ3c.so")
    
    # Read the compressed file into numpy array as uint8
    data_cmpr = np.fromfile(input_file, dtype=np.uint8)
    
    print(f"Decompressing: {os.path.basename(input_file)}")
    print(f"Compressed data shape: {data_cmpr.shape}, dtype: {data_cmpr.dtype}")

    # Decompress the data - pass numpy array, not file path
    data_decomp = sz.decompress(data_cmpr, shape, np.float32)
    print(f"Decompressed data shape: {data_decomp.shape}, dtype: {data_decomp.dtype}")
    print(f"Decompressed data range: [{np.min(data_decomp):.6e}, {np.max(data_decomp):.6e}]")

    # Save decompressed data
    data_decomp.tofile(output_file)
    print(f"Saved to: {output_file}\n")

    return data_decomp

def main():
    # Input and output directories
    input_dir = "/Users/923714256/Data_compression/SDRBENCH-EXASKY-NYX-512x512x512"
    output_dir = "/Users/923714256/Data_compression/data_compression_result"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of files to compress
    files = [
        "baryon_density.f32",
        "dark_matter_density.f32", 
        "temperature.f32",
        "velocity_x.f32",
        "velocity_y.f32",
        "velocity_z.f32"
    ]
    
    # Compression parameters
    eb_mode = 0  # 0 = A
    eb_abs = 0
    eb_rel = 5e-3
    eb_pwr = 0
    
    # Compress each file
    for filename in files:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename + ".sz3")
        
        if os.path.exists(input_file):
            compress_f32_file(input_file, output_file, 
                            shape=(512, 512, 512),
                            eb_mode=eb_mode, 
                            eb_abs=eb_abs,
                            eb_rel=eb_rel,
                            eb_pwr=eb_pwr)
            decompress_f32_file(output_file, reconstructed_file, 
                            shape=(512, 512, 512))
        else:
            print(f"Warning: {input_file} not found, skipping...")

def main():
    # Input and output directories
    input_dir = "/Users/923714256/Data_compression/SDRBENCH-EXASKY-NYX-512x512x512"
    output_dir = "/Users/923714256/Data_compression/data_compression_result"
    reconstructed_dir = "/Users/923714256/Data_compression/data_compression_result_reconstructed"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)
    # List of files to compress
    files = [
        "baryon_density.f32",
        "dark_matter_density.f32", 
        "temperature.f32",
        "velocity_x.f32",
        "velocity_y.f32",
        "velocity_z.f32"
    ]
    
    # Compression parameters
    eb_mode = 0  # 0 = A
    eb_abs = 0
    eb_rel = 5e-3
    eb_pwr = 0
    
    # Compress each file
    for filename in files:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename + ".sz")
        reconstructed_file = os.path.join(reconstructed_dir, filename + ".f32")
        if os.path.exists(input_file):
            compress_f32_file(input_file, output_file, 
                            shape=(512, 512, 512),
                            eb_mode=eb_mode, 
                            eb_abs=eb_abs,
                            eb_rel=eb_rel,
                            eb_pwr=eb_pwr)
            decompress_f32_file(output_file, reconstructed_file, 
                            shape=(512, 512, 512))
        else:
            print(f"Warning: {input_file} not found, skipping...")

if __name__ == "__main__":
    main()