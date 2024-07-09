from esm import pretrained
import torch

# Load the ESM2 model
esm2_model, alphabet = pretrained.esm2_t33_650M_UR50S()


class GraphTransformer(torch.nn.Module):
    def __init__(self, feature_size, esm_embedding_size, model_params):
        super(GraphTransformer, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

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
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))
            

        # Linear layers
        self.linear1 = Linear(embedding_size*2 + esm_embedding_size, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), 10)  # Output layer for 10 classes

    def forward(self, x, edge_attr, edge_index, batch_index, esm2_embeddings):
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
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                    )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
        x = sum(global_representation)

        # Concatenate ESM2 embeddings
        x = torch.cat((x, esm2_embeddings), dim=1)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x


# Example forward pass with ESM2 embeddings
def forward_pass(graph_transformer, x, edge_attr, edge_index, batch_index, sequence):
    with torch.no_grad():
        esm2_results = esm2_model(sequence, repr_layers=[33], return_contacts=False)
    esm2_embeddings = esm2_results["representations"][33]

    output = graph_transformer(x, edge_attr, edge_index, batch_index, esm2_embeddings)
    return output


criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(graph_transformer.parameters(), lr=0.001)

for epoch in range(num_epochs):
    graph_transformer.train()
    optimizer.zero_grad()
    
    output = forward_pass(graph_transformer, x, edge_attr, edge_index, batch_index, sequence)
    
    # Assuming labels are in the format required for multi-class classification
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
