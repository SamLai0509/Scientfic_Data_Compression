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


# ============================================================
# Skipping Sampling
# ============================================================


def create_strided_datasets(x_data, y_data, stride=4, spatial_dims=3,
                            val_split=0.1, shuffle=True, seed=42):
    """
    NeurLZ 风格的跳跃采样：从整个数据集中稀疏采样点。
    
    这与 patch 采样不同：
    - Patch 采样: 将数据切成不重叠的块
    - 跳跃采样: 每隔 stride 个点采样一个，保持全局分布
    
    Args:
        x_data: 输入数据 (H, W, D) 或 (N, H, W)
        y_data: 目标数据 (同形状)
        stride: 采样步长 (每隔 stride 个点取一个)
        spatial_dims: 2 或 3
        val_split: 验证集比例
        shuffle: 是否打乱
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset
    """
    np.random.seed(seed)
    
    if spatial_dims == 3:
        # 3D 跳跃采样
        h, w, d = x_data.shape
        
        # 生成采样网格
        h_indices = np.arange(0, h, stride)
        w_indices = np.arange(0, w, stride)
        d_indices = np.arange(0, d, stride)
        
        # 创建所有采样点的索引
        all_indices = []
        for hi in h_indices:
            for wi in w_indices:
                for di in d_indices:
                    all_indices.append((hi, wi, di))
        
        all_indices = np.array(all_indices)
        
    else:  # 2D
        n_slices, h, w = x_data.shape
        
        h_indices = np.arange(0, h, stride)
        w_indices = np.arange(0, w, stride)
        
        all_indices = []
        for slice_idx in range(n_slices):
            for hi in h_indices:
                for wi in w_indices:
                    all_indices.append((slice_idx, hi, wi))
        
        all_indices = np.array(all_indices)
    
    # 打乱并分割
    if shuffle:
        np.random.shuffle(all_indices)
    
    n_total = len(all_indices)
    n_train = int(n_total * (1 - val_split))
    
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    
    # 创建数据集
    train_dataset = StridedDataset(x_data, y_data, train_indices, spatial_dims)
    val_dataset = StridedDataset(x_data, y_data, val_indices, spatial_dims) if len(val_indices) > 0 else None
    
    return train_dataset, val_dataset


class StridedDataset:
    """
    跳跃采样数据集：存储采样点的索引，按需获取数据。
    
    与 PatchDataset 不同，这里存储的是点索引，不是预提取的 patch。
    """
    
    def __init__(self, x_data, y_data, indices, spatial_dims, context_size=7):
        """
        Args:
            x_data: 完整输入数据
            y_data: 完整目标数据
            indices: 采样点索引列表
            spatial_dims: 2 或 3
            context_size: 每个采样点周围的上下文窗口大小
        """
        self.x_data = x_data
        self.y_data = y_data
        self.indices = indices
        self.spatial_dims = spatial_dims
        self.context_size = context_size
        self.half_ctx = context_size // 2
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        返回采样点周围的上下文窗口。
        
        对于每个采样点 (h, w, d)，返回以该点为中心的 context_size³ 窗口。
        """
        if self.spatial_dims == 3:
            h, w, d = self.indices[idx]
            H, W, D = self.x_data.shape
            
            # 计算窗口边界（带边界检查）
            h_start = max(0, h - self.half_ctx)
            h_end = min(H, h + self.half_ctx + 1)
            w_start = max(0, w - self.half_ctx)
            w_end = min(W, w + self.half_ctx + 1)
            d_start = max(0, d - self.half_ctx)
            d_end = min(D, d + self.half_ctx + 1)
            
            # 提取上下文窗口
            x_patch = self.x_data[h_start:h_end, w_start:w_end, d_start:d_end]
            y_patch = self.y_data[h_start:h_end, w_start:w_end, d_start:d_end]
            
        else:  # 2D
            slice_idx, h, w = self.indices[idx]
            H, W = self.x_data.shape[1], self.x_data.shape[2]
            
            h_start = max(0, h - self.half_ctx)
            h_end = min(H, h + self.half_ctx + 1)
            w_start = max(0, w - self.half_ctx)
            w_end = min(W, w + self.half_ctx + 1)
            
            x_patch = self.x_data[slice_idx, h_start:h_end, w_start:w_end]
            y_patch = self.y_data[slice_idx, h_start:h_end, w_start:w_end]
        
        return x_patch, y_patch


def create_neurlz_style_datasets(x_data, y_data, spatial_dims=3,
                                  stride=4, context_size=7,
                                  val_split=0.1, seed=42):
    """
    完整的 NeurLZ 风格数据集创建函数。
    
    NeurLZ 的核心思想：
    1. 不是切分成大的 patch，而是稀疏采样点
    2. 每个采样点取一个小的上下文窗口
    3. 网络学习：上下文窗口 → 中心点的残差
    
    Args:
        x_data: SZ3 解压数据
        y_data: 残差 (原始 - 解压)
        spatial_dims: 2 或 3
        stride: 采样步长
        context_size: 上下文窗口大小（必须是奇数）
        val_split: 验证集比例
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset
    """
    assert context_size % 2 == 1, "context_size 必须是奇数"
    
    return create_strided_datasets(
        x_data, y_data, 
        stride=stride,
        spatial_dims=spatial_dims,
        val_split=val_split,
        shuffle=True,
        seed=seed
    )


# ============================================================
# 混合采样：结合 Patch 和 Strided 的优点
# ============================================================

def create_hybrid_datasets(x_data, y_data, patch_size=64, overlap=16,
                           spatial_dims=3, val_split=0.1, seed=42,
                           roi_mask=None, keep="all"):
    """
    Mixing Sampling: Using overlapping patches to reduce boundary effects.
    
    This is an improved version of Patch Sampling:
    - Allow patches to overlap
    - Reduce boundary discontinuity
    
    Args:
        x_data: Input data
        y_data: Target data
        patch_size: Patch size
        overlap: Overlap between patches
        spatial_dims: 2 or 3
        val_split: Validation set ratio
        seed: Random seed
        roi_mask: Optional boolean mask aligned with x_data/y_data.
                  - spatial_dims==2: shape (N, H, W)
                  - spatial_dims==3: shape (H, W, D)
        keep: "all" (no filtering), "roi" (keep patches that intersect roi_mask),
              "bg" (keep patches that do NOT intersect roi_mask)
    """
    np.random.seed(seed)
    
    stride = patch_size - overlap
    
    if keep not in ("all", "roi", "bg"):
        raise ValueError(f"keep must be one of ('all','roi','bg'), got: {keep}")

    # If no mask provided, fall back to original behavior
    if roi_mask is None or keep == "all":
        if spatial_dims == 3:
            all_patches_x, all_patches_y = _extract_3d_patches_with_overlap(
                x_data, y_data, patch_size, stride
            )
        else:
            all_patches_x, all_patches_y = _extract_2d_patches_with_overlap(
                x_data, y_data, patch_size, stride
            )
    else:
        # Mask-filtered extraction (keeps only ROI- or BG- patches)
        if spatial_dims == 3:
            all_patches_x, all_patches_y = _extract_3d_patches_with_overlap_masked(
                x_data, y_data, roi_mask, patch_size, stride, keep
            )
        else:
            all_patches_x, all_patches_y = _extract_2d_patches_with_overlap_masked(
                x_data, y_data, roi_mask, patch_size, stride, keep
            )
    
    # Split into train/val
    total = len(all_patches_x)
    indices = np.arange(total)
    np.random.shuffle(indices)
    
    n_train = int(total * (1 - val_split))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_patches_x = [all_patches_x[i] for i in train_idx]
    train_patches_y = [all_patches_y[i] for i in train_idx]
    train_dataset = PatchDataset(train_patches_x, train_patches_y, spatial_dims)
    
    if len(val_idx) > 0:
        val_patches_x = [all_patches_x[i] for i in val_idx]
        val_patches_y = [all_patches_y[i] for i in val_idx]
        val_dataset = PatchDataset(val_patches_x, val_patches_y, spatial_dims)
    else:
        val_dataset = None
    
    return train_dataset, val_dataset


def _extract_3d_patches_with_overlap_masked(x_data, y_data, roi_mask, patch_size, stride, keep):
    """Extract 3D patches with overlap, filtered by roi_mask intersection."""
    h, w, d = x_data.shape
    all_patches_x = []
    all_patches_y = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            for k in range(0, d - patch_size + 1, stride):
                m = roi_mask[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                has_roi = bool(np.any(m))
                if keep == "roi" and not has_roi:
                    continue
                if keep == "bg" and has_roi:
                    continue
                patch_x = x_data[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                patch_y = y_data[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                all_patches_x.append(patch_x.copy())
                all_patches_y.append(patch_y.copy())
    return all_patches_x, all_patches_y


def _extract_2d_patches_with_overlap_masked(x_data, y_data, roi_mask, patch_size, stride, keep):
    """Extract 2D patches with overlap, filtered by roi_mask intersection."""
    n_slices, h, w = x_data.shape
    all_patches_x = []
    all_patches_y = []
    for s in range(n_slices):
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                m = roi_mask[s, i:i+patch_size, j:j+patch_size]
                has_roi = bool(np.any(m))
                if keep == "roi" and not has_roi:
                    continue
                if keep == "bg" and has_roi:
                    continue
                patch_x = x_data[s, i:i+patch_size, j:j+patch_size]
                patch_y = y_data[s, i:i+patch_size, j:j+patch_size]
                all_patches_x.append(patch_x.copy())
                all_patches_y.append(patch_y.copy())
    return all_patches_x, all_patches_y


def _extract_3d_patches_with_overlap(x_data, y_data, patch_size, stride):
    """Extract 3D patches with overlap"""
    h, w, d = x_data.shape
    
    all_patches_x = []
    all_patches_y = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            for k in range(0, d - patch_size + 1, stride):
                patch_x = x_data[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                patch_y = y_data[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                
                all_patches_x.append(patch_x.copy())
                all_patches_y.append(patch_y.copy())
    
    return all_patches_x, all_patches_y


def _extract_2d_patches_with_overlap(x_data, y_data, patch_size, stride):
    """Extract 2D patches with overlap"""
    n_slices, h, w = x_data.shape
    
    all_patches_x = []
    all_patches_y = []
    
    for s in range(n_slices):
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch_x = x_data[s, i:i+patch_size, j:j+patch_size]
                patch_y = y_data[s, i:i+patch_size, j:j+patch_size]
                
                all_patches_x.append(patch_x.copy())
                all_patches_y.append(patch_y.copy())
    
    return all_patches_x, all_patches_y