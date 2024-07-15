import torch
import torch.nn.functional as F 
import esm
import torch.nn as nn

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

torch.manual_seed(42)

class GraphTransformer(torch.nn.Module):
    def __init__(self, feature_size, edge_dim, model_params):
        super(GraphTransformer, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
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
                                                    beta=True)) # enables a learnable scaling factor for attention scores.

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if self.use_pooling and i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.output_dim = embedding_size * 2

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(F.leaky_relu(self.transf1(x)))
        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.bn_layers[i](F.leaky_relu(self.transf_layers[i](x)))
                
            # Always aggregate last layer
            if self.use_pooling and (i % self.top_k_every_n == 0 or i == self.n_layers - 1):
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](x, edge_index, edge_attr, batch_index)
                    
            # Add current representation
            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
        x = sum(global_representation)
        return x

    # def forward(self, x, edge_attr, edge_index, batch_index):

    #     # Initial transformation
    #     x = self.conv1(x, edge_index, edge_attr)
    #     print(f"After conv1 - x shape: {x.shape}, var: {x.var()}")
    
    #     x = torch.relu(self.transf1(x))
    #     print(f"After transf1 and ReLU - x shape: {x.shape}, var: {x.var()}")
    
    #     x = self.bn1(x)
    #     print(f"After bn1 - x shape: {x.shape}, var: {x.var()}")
    
    #     # Holds the intermediate graph representations
    #     global_representation = []
    
    #     for i in range(self.n_layers):
    #         x = self.conv_layers[i](x, edge_index, edge_attr)
    #         print(f"Layer {i}, after conv - x shape: {x.shape}, var: {x.var()}")
    
    #         x = torch.relu(self.transf_layers[i](x))
    #         print(f"Layer {i}, after transf and ReLU - x shape: {x.shape}, var: {x.var()}")
    
    #         x = self.bn_layers[i](x)
    #         print(f"Layer {i}, after bn - x shape: {x.shape}, var: {x.var()}")
                
    #         # Always aggregate last layer
    #         if self.use_pooling and (i % self.top_k_every_n == 0 or i == self.n_layers - 1):
    #             x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](x, edge_index, edge_attr, batch_index)
    #             print(f"Layer {i}, after pooling - x shape: {x.shape}, var: {x.var()}")
                    
    #         # Add current representation
    #         current_repr = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
    #         print(f"Layer {i}, global representation shape: {current_repr.shape}, var: {current_repr.var()}")
    #         global_representation.append(current_repr)
    
    #     x = sum(global_representation)
    #     print(f"Final output shape: {x.shape}, var: {x.var()}")
    #     return x

class ESM2Model(torch.nn.Module):
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        super(ESM2Model, self).__init__()
        # check if the model files are already present in the local cache. 
        # If they're not found, it automatically downloads them from the specified URLs
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # Set to evaluation mode
        for param in self.model.parameters(): # Freeze ESM2 weights as its pre-trained
            param.requires_grad = False
        self.output_dim = self.model.embed_dim


    def forward(self, sequences):       
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        embeddings = results["representations"][self.model.num_layers]

        # Average pooling over sequence length to get one embedding per sequence
        # `esm2_repr` will have a shape of (batch_size, embedding_dim), 
        # which can be directly concatenated with the graph representations.
        sequence_embeddings = embeddings.mean(dim=1)
        
        return sequence_embeddings

class HybridModel(torch.nn.Module):
    def __init__(self, feature_size, edge_dim, model_params, num_classes=10):
        super(HybridModel, self).__init__()
        self.graph_transformer = GraphTransformer(feature_size, edge_dim, model_params) if model_params["use_graph"] else None
        self.esm2_model = ESM2Model() if model_params["use_esm2"] else None
        
        self.use_graph = model_params["use_graph"]
        self.use_esm2 = model_params["use_esm2"]

        # Store the dropout rate
        self.dropout_rate = model_params["model_dropout_rate"]

        # Calculate the input dimension for the classification layers
        input_dim = 0
        if self.use_graph:
            input_dim += self.graph_transformer.output_dim
        if self.use_esm2:
            input_dim += self.esm2_model.output_dim

        dense_neurons = model_params["model_dense_neurons"]

        # Linear layers with batch normalization
        self.linear1 = nn.Linear(input_dim, dense_neurons)
        self.bn1 = nn.BatchNorm1d(dense_neurons)
        self.linear2 = nn.Linear(dense_neurons, int(dense_neurons/2))
        self.bn2 = nn.BatchNorm1d(int(dense_neurons/2))
        self.linear3 = nn.Linear(int(dense_neurons/2), num_classes)

        # Linear layer to match dimensions for residual connection
        self.match_dim = nn.Linear(dense_neurons, int(dense_neurons/2))
        
        # # Linear layers
        # self.linear1 = Linear(input_dim, dense_neurons)
        # self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        # self.linear3 = Linear(int(dense_neurons/2), num_classes)  

    def forward(self, x, edge_attr, edge_index, batch_index, sequences):
        representations = []
        
        if self.use_graph:
            graph_repr = self.graph_transformer(x, edge_attr, edge_index, batch_index)
            representations.append(graph_repr)
            
        if self.use_esm2:
            esm2_repr = self.esm2_model(sequences)
            representations.append(esm2_repr)
        
        # Concatenate only the representations that are in use
        combined_repr = torch.cat(representations, dim=1)

        # Output block with batch normalization and residual connections
        x1 = self.bn1(torch.relu(self.linear1(combined_repr)))
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
        x1_residual = self.match_dim(x1)         # Match dimensions for residual connection

        x2 = self.bn2(torch.relu(self.linear2(x1)))
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
        x = self.linear3(x2 + x1_residual)  # Residual connection

        # # Output block
        # x = torch.relu(self.linear1(combined_repr))
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # x = torch.relu(self.linear2(x))
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # x = self.linear3(x)

        return x