import torch
from torch.utils.data import TensorDataset, DataLoader


def load_feature_matrix(dataset, device, batch_size):
    """
    Load dataset from .npz file and create DataLoader instances.
    """
    train_x = torch.tensor(dataset['train_x'], dtype=torch.float32).to(device)
    train_target = torch.tensor(dataset['train_target'], dtype=torch.float32).to(device)
    val_x = torch.tensor(dataset['val_x'], dtype=torch.float32).to(device)
    val_target = torch.tensor(dataset['val_target'], dtype=torch.float32).to(device)
    test_x = torch.tensor(dataset['test_x'], dtype=torch.float32).to(device)
    test_target = torch.tensor(dataset['test_target'], dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_x, train_target)
    val_dataset = TensorDataset(val_x, val_target)
    test_dataset = TensorDataset(test_x, test_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
