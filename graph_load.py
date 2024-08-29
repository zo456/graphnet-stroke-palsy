from torch_geometric.loader import DataLoader
import graph_data

def LoadGraphData(data_dir, batchsize_train, batchsize_val, split_id):

    #val_flag = (split_id + 1) % 3
    #test_flag = (split_id + 2) % 3
    train_dataset = graph_data.GraphDataset(
        root = data_dir,
        split=split_id,
        mode = 'train'
    )

    val_dataset = graph_data.GraphDataset(
        root = data_dir,
        split=split_id,
        mode = 'val'
    )

    test_dataset = graph_data.GraphDataset(
        root = data_dir,
        split=split_id,
        mode = 'test'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batchsize_train,
        shuffle = True,
        num_workers = 0,
        pin_memory = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batchsize_val,
        shuffle = True,
        num_workers = 0,
        pin_memory = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batchsize_val,
        shuffle = True,
        num_workers = 0,
        pin_memory = True
    )

    return train_loader, val_loader, test_loader

def TestData(data_dir, batch_size):
    test_dataset = graph_data.CrossDataset(
            root = data_dir
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle = True,
        num_workers = 0,
        pin_memory = True
    )

    return test_loader