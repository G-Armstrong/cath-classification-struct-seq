import torch.optim as optim
import numpy as np
import glob
import os
import torch.cuda.amp as amp
import torch.nn as nn
import math

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from model import *
from misc_utils import *
from torch.optim.lr_scheduler import LambdaLR

# def lr_lambda(epoch):
#     if epoch < 10:
#         return float(epoch) / 10
#     else:
#         return 1.0

def init_weights(m):
    if isinstance(m, TransformerConv):
        # print(f"Initializing TransformerConv layer")
        nn.init.xavier_uniform_(m.lin_key.weight)
        nn.init.xavier_uniform_(m.lin_query.weight)
        nn.init.xavier_uniform_(m.lin_value.weight)
        if m.lin_edge is not None:
            nn.init.xavier_uniform_(m.lin_edge.weight)
            if m.lin_edge.bias is not None:
                nn.init.zeros_(m.lin_edge.bias)
        
        nn.init.zeros_(m.lin_key.bias)
        nn.init.zeros_(m.lin_query.bias)
        nn.init.zeros_(m.lin_value.bias)
        
    elif isinstance(m, nn.Linear):
        # print(f"Initializing Linear layer: {m}")
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.BatchNorm1d):
        # print(f"Initializing BatchNorm1d layer: {m}")
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    else:
        initialized = False
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight)
            initialized = True   
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
            initialized = True    
        # if not initialized:
        #     print(f"Layer {m} not specifically initialized")

def run_ablation_study(train_data, val_data, test_data, model_params, num_classes, device):
    feature_size = train_data[0].x[0].shape[0]  # Automatically determine the number of node features
    edge_dim = train_data[0].edge_attr.shape[1]  # Number of edge features

    # configurations = [
    #     ("GraphTransformer", {"use_graph": True, "use_esm2": False}),
    #     ("ESM2", {"use_graph": False, "use_esm2": True}),
    #     ("Hybrid", {"use_graph": True, "use_esm2": True})
    # ]

    configurations = [
        ("GraphTransformer", {"use_graph": True, "use_esm2": False}),
        ("Hybrid", {"use_graph": True, "use_esm2": True})
    ]

    results = {}

    for config_name, config_params in configurations:
        print(f"\nRunning ablation study for: {config_name}")
        model_params.update(config_params)
        model = HybridModel(feature_size, edge_dim, model_params, num_classes)

        # Weight Initialization
        try:
            
            # Apply weight initialization based on the configuration
            if model.use_graph:
                model.graph_transformer.apply(init_weights)
            
            if model.use_esm2:
                # Ensure ESM2 model remains frozen
                for param in model.esm2_model.parameters():
                    param.requires_grad = False
                    
            # Always initialize the final linear layers
            model.linear1.apply(init_weights)
            model.linear2.apply(init_weights)
            model.linear3.apply(init_weights)
            model.match_dim.apply(init_weights)
            print('Model weights initialized.')
            
        except Exception as e:
            print(f"Error initializing model weights: {e}")
            print("Model structure:")
            print(model)
            raise

        # Loss and optimizer for general multi-class classo
        criterion = torch.nn.CrossEntropyLoss()
        # Weight decay is a form of L2 regularization. It adds a small penalty to the loss function for larger weights, 
        # which encourages the model to use smaller weights. This can help prevent overfitting.
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        test_loader = DataLoader(test_data, batch_size=32)

        train_losses, train_accs, val_losses, val_accs = train_model(model, config_name, train_loader, val_loader, 
                                                                     criterion, optimizer, num_epochs=50, device=device)
        # train_losses, train_accs, val_losses, val_accs = train_model_memory_efficient(model, config_name, train_loader, val_loader, 
        #                                                                               criterion, optimizer, num_epochs=50, device=device,
        #                                                                               accumulation_steps=10, use_amp=True)

        plot_training_curves(train_losses, train_accs, val_losses, val_accs, config_name)

        # Load best model checkpoint to perform evaluation
        checkpoint_path = f'states/checkpoint_best_{config_name}_model.pth'
        if os.path.exists(checkpoint_path):
            model, optimizer, epoch, best_val_acc = load_checkpoint(model, optimizer, checkpoint_path)
            model.to(device) # Ensure model is on the correct device after loading checkpoint
            print(f"Loaded best checkpoint for {config_name} from epoch {epoch} with validation accuracy {best_val_acc:.4f}")
        else:
            print(f"No checkpoint found for {config_name}, using the last trained model")

        # Evaluate performance
        test_loss, test_acc, precision, recall, f1, conf_matrix = evaluate_model(model, test_loader, criterion, device)
        # test_loss, test_acc, precision, recall, f1, conf_matrix = evaluate_model_memory_efficient(model, test_loader, criterion, device, use_amp=True)

        results[config_name] = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"\nResults for {config_name}:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        architecture_names = {
                                (1, 10): {"name": "Mainly Alpha: Orthogonal Bundle", "label": 0},
                                (1, 20): {"name": "Mainly Alpha: Up-down Bundle", "label": 1},
                                (2, 30): {"name": "Mainly Beta: Roll", "label": 2},
                                (2, 40): {"name": "Mainly Beta: Beta Barrel", "label": 3},
                                (2, 60): {"name": "Mainly Beta: Sandwich", "label": 4},
                                (3, 10): {"name": "Alpha Beta: Roll", "label": 5},
                                (3, 20): {"name": "Alpha Beta: Alpha-Beta Barrel", "label": 6},
                                (3, 30): {"name": "Alpha Beta: 2-Layer Sandwich", "label": 7},
                                (3, 40): {"name": "Alpha Beta: 3-Layer(aba) Sandwich", "label": 8},
                                (3, 90): {"name": "Alpha Beta: Alpha-Beta Complex", "label": 9}
                             }
        class_names = [str(key) for key in architecture_names.keys()]
        plot_confusion_matrix(conf_matrix, class_names, config_name)

    return results

def train_model(model, config_name, train_loader, val_loader, criterion, optimizer, num_epochs, device, 
                use_grad_clip=True, clip_value=2.0, patience=5, delta=0): # Change parameter values to experiment
    model.to(device)
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Early stopping variables
    best_score = 0.0
    counter = 0
    early_stop = False

    # # Learning rate scheduler with warmup
    # scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            sequences = batch.sequence
            labels = batch.y

            # Check for NaNs in input data (should set batch_size = 1 to use this debug code)
            # if torch.isnan(batch.x).any() or torch.isnan(batch.edge_attr).any() or torch.isnan(batch.edge_index).any():
            #     print(f"NaN detected in input data at batch {batch.cath_id[0][0]}")
            #     torch.save(batch, os.path.join('problematic_batches', f'{batch.cath_id[0][0]}.pt'))
            #     continue      
            
            optimizer.zero_grad()
            # `batch.batch` attribute is automatically created by PyTorch Geometric's DataLoader
            outputs = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch, sequences)
            # print(f'GT outputs: {outputs}')
            # print(f'GT labels: {labels}')

            loss = criterion(outputs, labels)
            loss.backward() # computes the gradients for all parameters.
             
            if use_grad_clip:
                # clips the gradients immediately after they've been computed
                clip_grad_norm_(model.parameters(), max_norm=clip_value)

            optimizer.step() # updates the parameters using the clipped gradients if use_grad_clip=True

            train_loss += loss.item()   
        
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Clear memory
            del batch, sequences, labels, outputs, loss
            torch.cuda.empty_cache()

        # Plot gradient flow after each epoch
        plot_grad_flow(model.named_parameters(), config_name, epoch + 1)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                val_sequences = val_batch.sequence
                val_labels = val_batch.y
                
                outputs = model(val_batch.x, val_batch.edge_attr, val_batch.edge_index, val_batch.batch, val_sequences)
                # print(f'GT VAL outputs: {outputs}')
                # print(f'GT VAL labels: {val_labels}')
                loss = criterion(outputs, val_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (predicted == val_labels).sum().item()

                # # Clear memory
                del val_batch, val_sequences, val_labels, outputs, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # # Step the scheduler and print current learning rate
        # scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Current learning rate: {current_lr}")
            
        # Early stopping logic
        if val_acc > best_score + delta:
            best_score = val_acc
            counter = 0
            save_checkpoint(model, optimizer, epoch+1, val_acc, f'states/checkpoint_best_{config_name}_model.pth')
            print(f'Validation accuracy increased ({best_val_acc:.6f} --> {val_acc:.6f}). Saving model ...')
            best_val_acc = val_acc
        else:
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {patience}')
            if counter >= patience:
                print("Early stopping")
                early_stop = True
                break
                
        # Clear memory after each epoch
        torch.cuda.empty_cache()

        if early_stop:
            print("Training stopped early.")
            break

    return train_losses, train_accs, val_losses, val_accs

def train_model_memory_efficient(model, config_name, train_loader, val_loader, criterion, optimizer, num_epochs, device, 
                                 accumulation_steps=4, use_amp=True):
    model.to(device)
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    scaler = amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch = batch.to(device)
            sequences = batch.sequence
            labels = batch.y

            with amp.autocast(enabled=use_amp):
                outputs = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch, sequences)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Clear memory
            del batch, sequences, labels, outputs, loss
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        # For validation and test data, we don't use gradient accumulation because we're not performing backpropagation or updating weights.
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                val_sequences = val_batch.sequence
                val_labels = val_batch.y
                
                with amp.autocast(enabled=use_amp):
                    outputs = model(val_batch.x, val_batch.edge_attr, val_batch.edge_index, val_batch.batch, val_sequences)
                    loss = criterion(outputs, val_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (predicted == val_labels).sum().item()

                # Clear memory
                del val_batch, val_sequences, val_labels, outputs, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, f'states/checkpoint_best_{config_name}_model.pth')

        # Clear memory after each epoch
        torch.cuda.empty_cache()

    return train_losses, train_accs, val_losses, val_accs

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            sequences = batch.sequence
            labels = batch.y
            
            outputs = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch, sequences)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return test_loss, test_acc, precision, recall, f1, conf_matrix

def evaluate_model_memory_efficient(model, test_loader, criterion, device, use_amp=True):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            sequences = batch.sequence
            labels = batch.y
            
            with amp.autocast(enabled=use_amp):
                outputs = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch, sequences)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Clear memory
            del batch, sequences, labels, outputs, loss
            torch.cuda.empty_cache()

    test_loss /= len(test_loader)
    test_acc = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return test_loss, test_acc, precision, recall, f1, conf_matrix