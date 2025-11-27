from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz
import numpy as np
import torch
import os


class Vectorized_Dataset(Dataset):
    """Dataset class for vectorized text data"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def scipy_to_torch_sparse(sparse_matrix):
    """Convert a scipy sparse matrix to a torch sparse tensor"""
    sparse_coo = sparse_matrix.tocoo()
    values = sparse_coo.data
    indices = np.vstack((sparse_coo.row, sparse_coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = torch.Size(sparse_coo.shape)
    
    return torch.sparse_coo_tensor(i, v, shape)


def load_vectorized_datasets(base_path, encoding_types=None, splits=None):
    """
    Load all vectorized datasets from .npz files
    
    Parameters:
    -----------
    base_path : str
        Path to the folder containing .npz files
    encoding_types : list, optional
        List of encoding types to load. Default: ['bin', 'freq', 'hash', 'tfidf']
    splits : list, optional
        List of splits to load. Default: ['train', 'val', 'test']
    
    Returns:
    --------
    dict
        Nested dictionary: data[encoding_type][split] = scipy_sparse_matrix
    """
    
    if encoding_types is None:
        encoding_types = ['bin', 'freq', 'hash', 'tfidf']
    
    if splits is None:
        splits = ['train', 'val', 'test']
    
    data = {}
    
    for enc_type in encoding_types:
        data[enc_type] = {}
        for split in splits:
            filename = f"data_{enc_type}_{split}.npz"
            filepath = os.path.join(base_path, filename)
            
            try:
                data[enc_type][split] = load_npz(filepath)
                print(f" Loaded: {filename} - Shape: {data[enc_type][split].shape}")
            except FileNotFoundError:
                print(f" File not found: {filename}")
            except Exception as e:
                print(f" Error loading {filename}: {e}")
    
    return data


def convert_to_torch_sparse(data_scipy):
    """
    Convert all scipy sparse matrices to torch sparse tensors
    
    Parameters:
    -----------
    data_scipy : dict
        Nested dictionary with scipy sparse matrices
    
    Returns:
    --------
    dict
        Nested dictionary with torch sparse tensors
    """
    
    data_torch = {}
    
    for enc_type, splits in data_scipy.items():
        data_torch[enc_type] = {}
        for split_name, sparse_matrix in splits.items():
            data_torch[enc_type][split_name] = scipy_to_torch_sparse(sparse_matrix)
    
    print("Converted all data to torch sparse tensors")
    return data_torch


def densify_data(data_sparse):
    """
    Convert all sparse tensors to dense tensors
    
    Parameters:
    -----------
    data_sparse : dict
        Nested dictionary with torch sparse tensors
    
    Returns:
    --------
    dict
        Nested dictionary with torch dense tensors
    """
    
    data_dense = {}
    
    for enc_type, splits in data_sparse.items():
        data_dense[enc_type] = {}
        for split_name, sparse_tensor in splits.items():
            data_dense[enc_type][split_name] = sparse_tensor.to_dense()
    
    print("Converted all data to dense tensors")
    return data_dense


def create_dataloaders(data_dense, encoding_type, y_train, y_val, y_test, batch_size=64):
    """
    Create DataLoaders for a specific encoding type
    
    Parameters:
    -----------
    data_dense : dict
        Nested dictionary with dense torch tensors
    encoding_type : str
        Type of encoding: 'bin', 'freq', 'hash', or 'tfidf'
    y_train, y_val, y_test : array-like
        Labels for each split
    batch_size : int
        Batch size for DataLoader
    
    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader)
    """
    
    # Convert labels to tensors if needed
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.LongTensor(y_train)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.LongTensor(y_val)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = Vectorized_Dataset(data_dense[encoding_type]['train'], y_train)
    val_dataset = Vectorized_Dataset(data_dense[encoding_type]['val'], y_val)
    test_dataset = Vectorized_Dataset(data_dense[encoding_type]['test'], y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created DataLoaders for {encoding_type}: "
          f"{len(train_loader)} train batches, "
          f"{len(val_loader)} val batches, "
          f"{len(test_loader)} test batches")
    
    return train_loader, val_loader, test_loader