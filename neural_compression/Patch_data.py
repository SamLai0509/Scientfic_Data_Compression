import numpy as np
import torch


class PatchDataset:
    """
    Dataset class that stores 2D or 3D patches/chunks.
    Similar to torch.utils.data.Dataset but optimized for our use case.
    """
    
    def __init__(self, patches_x, patches_y, spatial_dims):
        """
        Args:
            patches_x: List of input patches (numpy arrays)
            patches_y: List of target patches (numpy arrays)
            spatial_dims: 2 or 3 for spatial dimensions
        """
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.spatial_dims = spatial_dims
        
    def __len__(self):
        return len(self.patches_x)
    
    def __getitem__(self, idx):
        """Return a single patch as (x, y) tuple."""
        return self.patches_x[idx], self.patches_y[idx]


class PatchDataLoader:
    """
    DataLoader for patch-based training with batching support.
    Handles both 2D patches and 3D chunks.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        Args:
            dataset: PatchDataset instance
            batch_size: Number of patches per batch
            shuffle: Whether to shuffle data at each iteration
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.spatial_dims = dataset.spatial_dims
        
    def __iter__(self):
        """Iterate over batches."""
        n = len(self.dataset)
        indices = np.arange(n)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue
            
            batch_indices = indices[start_idx:end_idx]
            
            # Collect patches in this batch
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            yield batch_x, batch_y
    
    def __len__(self):
        """Return number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        else:
            return (n + self.batch_size - 1) // self.batch_size


def create_patch_datasets(x_data, y_data, patch_size=32, spatial_dims=3, 
                          val_split=0.1, shuffle=True, seed=42):
    """
    Create train and validation patch datasets from input data.
    
    Args:
        x_data: Input data (numpy array)
            - For 2D: (N, H, W) - N slices
            - For 3D: (H, W, D) - volume
        y_data: Target data (same shape as x_data)
        patch_size: Size of patches/chunks (int or tuple)
        spatial_dims: 2 for 2D patches, 3 for 3D chunks
        val_split: Fraction of patches for validation (0.0-1.0)
        shuffle: Whether to shuffle patches before splitting
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset: PatchDataset instances
    """
    np.random.seed(seed)
    
    # Convert patch_size to tuple
    if isinstance(patch_size, int):
        if spatial_dims == 2:
            patch_size = (patch_size, patch_size)
        else:
            patch_size = (patch_size, patch_size, patch_size)
    
    # Extract patches
    if spatial_dims == 2:
        all_patches_x, all_patches_y = _extract_2d_patches(x_data, y_data, patch_size)
    else:
        all_patches_x, all_patches_y = _extract_3d_patches(x_data, y_data, patch_size)
    
    # Split into train/val
    total_patches = len(all_patches_x)
    indices = np.arange(total_patches)
    
    if shuffle:
        np.random.shuffle(indices)
    
    n_train = int(total_patches * (1 - val_split))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_patches_x = [all_patches_x[i] for i in train_indices]
    train_patches_y = [all_patches_y[i] for i in train_indices]
    train_dataset = PatchDataset(train_patches_x, train_patches_y, spatial_dims)
    
    if len(val_indices) > 0:
        val_patches_x = [all_patches_x[i] for i in val_indices]
        val_patches_y = [all_patches_y[i] for i in val_indices]
        val_dataset = PatchDataset(val_patches_x, val_patches_y, spatial_dims)
    else:
        val_dataset = None
    
    return train_dataset, val_dataset


def _extract_2d_patches(x_data, y_data, patch_size):
    """
    Extract 2D patches from sliced data.
    Input shape: (N, H, W)
    Returns: Lists of patches
    """
    n_slices, h, w = x_data.shape
    patch_h, patch_w = patch_size
    
    # Calculate number of patches per slice
    n_patches_h = (h + patch_h - 1) // patch_h
    n_patches_w = (w + patch_w - 1) // patch_w
    
    # Collect all patches
    all_patches_x = []
    all_patches_y = []
    
    for slice_idx in range(n_slices):
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                h_start = i * patch_h
                h_end = min((i + 1) * patch_h, h)
                w_start = j * patch_w
                w_end = min((j + 1) * patch_w, w)
                
                patch_x = x_data[slice_idx, h_start:h_end, w_start:w_end]
                patch_y = y_data[slice_idx, h_start:h_end, w_start:w_end]
                
                all_patches_x.append(patch_x)
                all_patches_y.append(patch_y)
    
    return all_patches_x, all_patches_y


def _extract_3d_patches(x_data, y_data, patch_size):
    """
    Extract 3D chunks from volume data.
    Input shape: (H, W, D)
    Returns: Lists of chunks
    """
    h, w, d = x_data.shape
    patch_h, patch_w, patch_d = patch_size
    
    # Calculate number of chunks in each dimension
    n_chunks_h = (h + patch_h - 1) // patch_h
    n_chunks_w = (w + patch_w - 1) // patch_w
    n_chunks_d = (d + patch_d - 1) // patch_d
    
    # Collect all chunks
    all_chunks_x = []
    all_chunks_y = []
    
    for i in range(n_chunks_h):
        for j in range(n_chunks_w):
            for k in range(n_chunks_d):
                h_start = i * patch_h
                h_end = min((i + 1) * patch_h, h)
                w_start = j * patch_w
                w_end = min((j + 1) * patch_w, w)
                d_start = k * patch_d
                d_end = min((k + 1) * patch_d, d)
                
                chunk_x = x_data[h_start:h_end, w_start:w_end, d_start:d_end]
                chunk_y = y_data[h_start:h_end, w_start:w_end, d_start:d_end]
                
                all_chunks_x.append(chunk_x)
                all_chunks_y.append(chunk_y)
    
    return all_chunks_x, all_chunks_y


def collate_patches_to_tensor(batch_x, batch_y, device='cuda', spatial_dims=3):
    """
    Convert a batch of numpy patches to PyTorch tensors.
    
    Args:
        batch_x: List of input patches (numpy arrays)
        batch_y: List of target patches (numpy arrays)
        device: Device to move tensors to
        spatial_dims: 2 or 3
    
    Returns:
        x_tensor, y_tensor: Batched tensors with shape:
            - 2D: (batch_size, 1, H, W)
            - 3D: (batch_size, 1, H, W, D)
    """
    # Stack patches into batch
    # Find target shape (max along each axis)
    def _max_shape(lst):
        shp = list(lst[0].shape)
        for a in lst[1:]:
            shp = [max(s, t) for s, t in zip(shp, a.shape)]
        return tuple(shp)
    def _pad_to(arr, tgt):
        pad = []
        for s, t in zip(arr.shape, tgt):
            diff = t - s
            # (left, right) padding per dimension – pad at the end
            pad.extend([0, max(0, diff)])
        # numpy.pad expects pad widths from last axis → reverse pairs
        pad = pad[::-1]
        return np.pad(arr, [(0, pad[i]) for i in range(0, len(pad), 2)][::-1], mode='edge')

    tgt_x = _max_shape(batch_x)
    tgt_y = _max_shape(batch_y)
    x = np.stack([_pad_to(a, tgt_x) for a in batch_x], axis=0)
    y = np.stack([_pad_to(a, tgt_y) for a in batch_y], axis=0)
 
    if spatial_dims == 2:
        x = torch.from_numpy(x).float().unsqueeze(1).to(device)  # (N,1,H,W)
        y = torch.from_numpy(y).float().unsqueeze(1).to(device)
    else:
         x = torch.from_numpy(x).float().unsqueeze(1).to(device)  # (N,1,D,H,W)
         y = torch.from_numpy(y).float().unsqueeze(1).to(device)
    return x, y


# Convenience function to get info
def get_dataset_info(train_dataset, val_dataset, patch_size, spatial_dims):
    """Return formatted dataset information."""
    if spatial_dims == 2:
        patch_type = "patches"
        if isinstance(patch_size, tuple):
            patch_shape = f"{patch_size[0]}×{patch_size[1]}"
        else:
            patch_shape = f"{patch_size}×{patch_size}"
    else:
        patch_type = "chunks"
        if isinstance(patch_size, tuple):
            patch_shape = f"{patch_size[0]}×{patch_size[1]}×{patch_size[2]}"
        else:
            patch_shape = f"{patch_size}×{patch_size}×{patch_size}"
    
    info = f"{spatial_dims}D {patch_type} ({patch_shape}): "
    info += f"Train={len(train_dataset)}"
    if val_dataset is not None:
        info += f", Val={len(val_dataset)}"
    
    return info