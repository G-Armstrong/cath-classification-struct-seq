import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

class GraphTransformer(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GraphTransformer, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]
        self.use_pooling = model_params["use_pooling"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size, 
                                    embedding_size, 
                                    heads=n_heads, 
                                    dropout=dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True) 

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size, 
                                                    embedding_size, 
                                                    heads=n_heads, 
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if self.use_pooling and i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.output_dim = embedding_size * 2

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if self.use_pooling and (i % self.top_k_every_n == 0 or i == self.n_layers - 1):
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                    )
            # Add current representation
            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
        x = sum(global_representation)
        return x

class ESM2Model(torch.nn.Module):
    def __init__(self, esm2_dim):
        super(ESM2Model, self).__init__()
        self.output_dim = esm2_dim

    def forward(self, esm2_embedding):
        return esm2_embedding

class CombinedModel(torch.nn.Module):
    def __init__(self, feature_size, model_params, esm2_dim, num_classes=10):
        super(CombinedModel, self).__init__()
        self.graph_transformer = GraphTransformer(feature_size, model_params)
        self.esm2_model = ESM2Model(esm2_dim)
        
        combined_dim = self.graph_transformer.output_dim + self.esm2_model.output_dim
        dense_neurons = model_params["model_dense_neurons"]

        self.use_esm2 = model_params["use_esm2"]
        self.use_graph = model_params["use_graph"]

        # Linear layers
        self.linear1 = Linear(combined_dim, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), num_classes)  

    def forward(self, x, edge_attr, edge_index, batch_index, esm2_embedding):
        graph_repr = self.graph_transformer(x, edge_attr, edge_index, batch_index) if self.use_graph else torch.zeros(esm2_embedding.shape[0], self.graph_transformer.output_dim).to(esm2_embedding.device)
        esm2_repr = self.esm2_model(esm2_embedding) if self.use_esm2 else torch.zeros(esm2_embedding.shape[0], self.esm2_model.output_dim).to(esm2_embedding.device)
        
        combined_repr = torch.cat([graph_repr, esm2_repr], dim=1)

        # Output block
        x = torch.relu(self.linear1(combined_repr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)

        return x

# Usage example
model_params = {
    "model_embedding_size": 64,
    "model_attention_heads": 8,
    "model_layers": 4,
    "model_dropout_rate": 0.2,
    "model_top_k_ratio": 0.5,
    "model_top_k_every_n": 2,
    "model_dense_neurons": 128,
    "model_edge_dim": 4,
    "use_pooling": True,
    "use_esm2": True,
    "use_graph": True
}

feature_size = 32  # Example value
esm2_dim = 1280  # ESM2 embedding dimension
num_classes = 10

model = CombinedModel(feature_size, model_params, esm2_dim, num_classes)

# Loss function for multi-class classification
criterion = torch.nn.CrossEntropyLoss()
