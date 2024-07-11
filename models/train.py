from model_utils import *

class ProteinDataset(Dataset):
    # Data(cath_id=[1], seq=[1], node_labels=[3341], node_feat=[3341, 8], edge_feat=[3384, 2], adj_matrix=[3341, 3341], target=0)
    def __init__(self, base_path, partition):
        self.base_path = base_path
        self.partition = partition
        self.data_list = self.load_data_from_partitions()
       
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Create a focused PyG Data object for the graph data only
        graph_data_batch = Data(cath_id=data.cath_id,
                                x=data.node_feat,
                                edge_attr=data.edge_feat,
                                edge_index=data.adj_matrix,    # Sparse Adj Matrix
                                y=data.target,                 # The target is our label
                                sequence=data.seq[0])          # seq is a list with one string element representing the raw AA sequence
       
        return graph_data_batch
        
    def load_data_from_partitions(self):
        partition_files = glob.glob(os.path.join(self.base_path, '*.pt'))
        data_list = []
        for partition_file in partition_files:
            partition_data = torch.load(partition_file)
            data_list.extend(partition_data)
            if self.partition == 'train': 
                if len(data_list) >= 1924: # Kill the loading of training data after this point to reduce RAM load
                    break
        
        print(f"[{self.partition.upper()}] Loaded {len(data_list)} PyG objects from {self.base_path}")
        return data_list

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Load data 
    print('Loading data...')
    train_val_dataset = ProteinDataset('../models/train_partitions', 'train')
    train_data, val_data = split_dataset(train_val_dataset, val_split=0.2)
    test_data = ProteinDataset('../models/test_partitions', 'test')
    print('Done.')
    
    model_params = {
        "model_embedding_size": 32,
        "model_attention_heads": 2,
        "model_layers": 4,
        "model_dropout_rate": 0.20,
        "model_top_k_ratio": 0.5,
        "model_top_k_every_n": 2 ,
        "model_dense_neurons": 64,
        "use_pooling": True
    }

    num_classes = 10  # Set this to your number of classes

    results = run_ablation_study(train_data, val_data, test_data, model_params, num_classes, device)

    # Print final comparison
    print("\nFinal Comparison:")
    for config, metrics in results.items():
        print(f"\n{config}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")