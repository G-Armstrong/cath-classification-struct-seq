import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import esm
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
        temp = x
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        if torch.isnan(x).any():
            print("NaN detected after operation conv1")
            print(f'Node_Feats:\n {temp} \n {temp.shape}')
            print(f'Edge_Feats:\n {edge_attr} \n {edge_attr.shape}')
            print(f'Edge_Index:\n {edge_index} \n {edge_index.shape}')
            raise ValueError("NaN in forward pass")
        x = torch.relu(self.transf1(x))
        if torch.isnan(x).any():
            print("NaN detected after operation transf1")
            raise ValueError("NaN in forward pass")
        x = self.bn1(x)
        if torch.isnan(x).any():
            print("NaN detected after operation bn1")
            raise ValueError("NaN in forward pass")

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            if torch.isnan(x).any():
                print("NaN detected after operation conv_layers[i]")
                raise ValueError("NaN in forward pass")
                
            x = torch.relu(self.transf_layers[i](x))
            if torch.isnan(x).any():
                print("NaN detected after operation transf_layers[i]")
                raise ValueError("NaN in forward pass")
                
            x = self.bn_layers[i](x)
            if torch.isnan(x).any():
                print("NaN detected after operation bn_layers[i]")
                raise ValueError("NaN in forward pass")
                
            # Always aggregate last layer
            if self.use_pooling and (i % self.top_k_every_n == 0 or i == self.n_layers - 1):
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](x, edge_index, edge_attr, batch_index)
                if torch.isnan(x).any():
                    print("NaN detected after operation pooling_layers[i]")
                    raise ValueError("NaN in forward pass")
                    
            # Add current representation
            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
        x = sum(global_representation)
        if torch.isnan(x).any():
            print("NaN detected after operation sum()")
            raise ValueError("NaN in forward pass")
        return x

class ESM2Model(torch.nn.Module):
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        super(ESM2Model, self).__init__()
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # Set to evaluation mode
        for param in self.model.parameters(): # Freeze ESM2 weights as its pre-trained
            param.requires_grad = False
        self.output_dim = self.model.args.embed_dim

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
        
        return embeddings

class HybridModel(torch.nn.Module):
    def __init__(self, feature_size, edge_dim, model_params, num_classes=10):
        super(HybridModel, self).__init__()
        self.graph_transformer = GraphTransformer(feature_size, edge_dim, model_params) if model_params["use_graph"] else None
        self.esm2_model = ESM2Model() if model_params["use_esm2"] else None
        
        self.use_graph = model_params["use_graph"]
        self.use_esm2 = model_params["use_esm2"]

        # Calculate the input dimension for the classification layers
        input_dim = 0
        if self.use_graph:
            input_dim += self.graph_transformer.output_dim
        if self.use_esm2:
            input_dim += self.esm2_model.output_dim

        dense_neurons = model_params["model_dense_neurons"]

        # Linear layers
        self.linear1 = Linear(input_dim, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), num_classes)  

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

        # Output block
        x = torch.relu(self.linear1(combined_repr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)

        return x