import torch 
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import json

num_layers = 2
batch_size = 32
epochs = 20
lr = 0.0001
tcher_force = 0.5
optimizer_name = 'adam'
hidden_dim = 512
embed_dim = 128
pad_index = 0

class MazeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, vocab):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_seq = eval(row['input_sequence'])
        target_seq = eval(row['output_path'])

        input_encoding = self.vocab.encode_sequence(input_seq)
        target_encoding = self.vocab.encode_sequence(target_seq)

        return torch.tensor(input_encoding), torch.tensor(target_encoding)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([len(seq) for seq in inputs])
    target_lengths = torch.tensor([len(seq) for seq in targets])

    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_index)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_index)

    return padded_inputs, padded_targets, input_lengths, target_lengths


class RNNModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.embed = nn.Embedding(input_dim, embed_dim)
        self.encoder_rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Bahdanau attention
        self.w_s = nn.Linear(hidden_dim, hidden_dim)
        self.w_h = nn.Linear(hidden_dim, hidden_dim)
        self.const_bahdanau = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim, input_dim)

    def encode(self, input, input_size):
        embed_input = self.embed(input)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embed_input, input_size.cpu(), batch_first=True, enforce_sorted=False
        )     
        packed_output, last_hidden = self.encoder_rnn(packed_input) 
        outputs = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)[0] 
        return outputs, last_hidden
    
    def bahdanau(self, last_hidden, all_hidden, mask):
        last_hidden = last_hidden.unsqueeze(1)
        eij = self.const_bahdanau(
            torch.tanh(self.w_h(all_hidden) + self.w_s(last_hidden))
        ).squeeze(2)
        eij = eij.masked_fill(mask == 0, -1e10)
        alpha_ij = F.softmax(eij, dim=1)
        c_i = torch.bmm(alpha_ij.unsqueeze(1), all_hidden).squeeze(1)
        return c_i
    
    def decode(self, input, all_hidden, state_hidden, mask):
        prev_state_hidden = state_hidden[-1]
        c_i = self.bahdanau(prev_state_hidden, all_hidden, mask)
        embed_input = self.embed(input)
        context_added_input = torch.cat((embed_input, c_i), dim=1).unsqueeze(1)
        output, state_hidden = self.decoder_rnn(context_added_input, state_hidden)
        output = output.squeeze(1)
        pred = self.out(output)
        return pred, state_hidden
    
    def forward(self, input, input_length, target, teacher_forcing=True):
        outputs = torch.zeros(input.size(0), target.size(1), self.input_dim).to(input.device)
        
        encoder_outputs, hidden = self.encode(input, input_length)
        mask = (input != pad_index).to(input.device)
        decoder_input = target[:, 0]
        decoder_hidden = hidden

        for t in range(1, target.size(1)):
            decoder_output, decoder_hidden = self.decode(
                decoder_input, encoder_outputs, decoder_hidden, mask
            )
            outputs[:, t, :] = decoder_output
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_forcing else top1

        return outputs


class Vocab:
    def __init__(self):
        temp = [f'({i},{j})' for i in range(6) for j in range(6)]
        num_tokens = 0
        self.token_to_index = {
            '<PAD>': 0, 
            '<ADJLIST_START>': 1, 
            '<ADJLIST_END>': 2, 
            '<ORIGIN_START>': 3, 
            '<ORIGIN_END>': 4, 
            '<TARGET_START>': 5, 
            '<TARGET_END>': 6,
            '<PATH_START>': 7,
            '<PATH_END>': 8,
            '<-->': 9, 
            ';': 10
        }
        num_tokens += len(self.token_to_index)
        for i in temp:
            self.token_to_index[i] = num_tokens
            num_tokens += 1

        self.vocab = list(self.token_to_index.keys())
        self.num_tokens = num_tokens
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}

    def encode_sequence(self, sequence):
        return [self.token_to_index[token] for token in sequence]
    
    def decode_sequence(self, indices):
        return [self.vocab[index] for index in indices]


def calculate_f1_score(predictions, targets, pad_idx=0):
    """
    Calculate F1 score for token-level predictions
    predictions: (batch * seq_len,) - flattened predictions
    targets: (batch * seq_len,) - flattened targets
    """
    # Remove padding
    mask = (targets != pad_idx)
    predictions = predictions[mask]
    targets = targets[mask]
    
    # Calculate true positives, false positives, false negatives
    # For each unique token (treating this as multi-class)
    unique_tokens = torch.unique(torch.cat([predictions, targets]))
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for token in unique_tokens:
        if token == pad_idx:
            continue
        tp = ((predictions == token) & (targets == token)).sum().item()
        fp = ((predictions == token) & (targets != token)).sum().item()
        fn = ((predictions != token) & (targets == token)).sum().item()
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate micro-averaged F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


def train_for_one_epoch(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    correct_tokens = 0
    total_tokens = 0
    all_predictions = []
    all_targets = []

    for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Teacher forcing decision per batch
        if random.random() < teacher_forcing_ratio:
            outputs = model(inputs, input_lengths, targets, teacher_forcing=True)
        else:
            outputs = model(inputs, input_lengths, targets, teacher_forcing=False)
        
        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
        targets_flat = targets[:, 1:].reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Calculate accuracy
        predictions = outputs_flat.argmax(1)
        mask = (targets_flat != 0)
        correct_tokens += ((predictions == targets_flat) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        # Store for F1 calculation
        all_predictions.append(predictions.cpu())
        all_targets.append(targets_flat.cpu())
    
    # Calculate F1 score
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    f1 = calculate_f1_score(all_predictions, all_targets)
    
    return epoch_loss / len(dataloader), correct_tokens / total_tokens, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model with all metrics"""
    model.eval()
    epoch_loss = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            
            outputs = model(inputs, input_lengths, targets, teacher_forcing=False)
            
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            targets_flat = targets[:, 1:].reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            epoch_loss += loss.item()
            
            # Token accuracy
            predictions = outputs_flat.argmax(1)
            mask = (targets_flat != 0)
            correct_tokens += ((predictions == targets_flat) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            # Store for F1
            all_predictions.append(predictions.cpu())
            all_targets.append(targets_flat.cpu())
            
            # Sequence accuracy
            batch_predictions = outputs.argmax(2)[:, 1:]
            batch_targets = targets[:, 1:]
            
            for pred, target, length in zip(batch_predictions, batch_targets, target_lengths):
                pred = pred[:length - 1]
                target = target[:length - 1]
                if torch.all(pred == target):
                    correct_sequences += 1
                total_sequences += 1
    
    # Calculate F1 score
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    f1 = calculate_f1_score(all_predictions, all_targets)
    
    return (epoch_loss / len(dataloader), 
            correct_tokens / total_tokens,
            correct_sequences / total_sequences,
            f1)


def get_random_predictions(model, dataset, vocab, device, num_samples=5):
    """Get predictions for random samples from dataset"""
    model.eval()
    
    # Select random indices
    indices = random.sample(range(len(dataset)), num_samples)
    predictions_list = []
    
    with torch.no_grad():
        for idx in indices:
            input_encoded, target_encoded = dataset[idx]
            
            # Prepare batch of size 1
            input_seq = input_encoded.unsqueeze(0).to(device)
            target_seq = target_encoded.unsqueeze(0).to(device)
            input_length = torch.tensor([len(input_encoded)]).to(device)
            
            # Get prediction
            outputs = model(input_seq, input_length, target_seq, teacher_forcing=False)
            predicted_indices = outputs.argmax(2)[0].cpu().numpy()
            
            # Decode sequences
            input_tokens = vocab.decode_sequence(input_encoded.tolist())
            target_tokens = vocab.decode_sequence(target_encoded.tolist())
            predicted_tokens = vocab.decode_sequence(predicted_indices.tolist())
            
            predictions_list.append({
                'input_sequence': input_tokens,
                'target_sequence': target_tokens,
                'predicted_sequence': predicted_tokens
            })
    
    return predictions_list


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot all training curves"""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs_range, history['test_loss'], 'r-', label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Token Accuracy
    axes[0, 1].plot(epochs_range, history['train_token_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs_range, history['test_token_acc'], 'r-', label='Test')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Token Accuracy')
    axes[0, 1].set_title('Token Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Sequence Accuracy (only test)
    axes[0, 2].plot(epochs_range, history['test_seq_acc'], 'r-', label='Test')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Sequence Accuracy')
    axes[0, 2].set_title('Sequence Accuracy (Test)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # F1 Score
    axes[1, 0].plot(epochs_range, history['train_f1'], 'b-', label='Train')
    axes[1, 0].plot(epochs_range, history['test_f1'], 'r-', label='Test')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Sequence Accuracy Percentage
    axes[1, 1].plot(epochs_range, [acc * 100 for acc in history['test_seq_acc']], 'r-', label='Test')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Sequence Accuracy (%)')
    axes[1, 1].set_title('Sequence Accuracy Percentage (Test)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Combined F1 and Seq Acc
    ax2 = axes[1, 2]
    ax2.plot(epochs_range, history['test_f1'], 'b-', label='F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_range, [acc * 100 for acc in history['test_seq_acc']], 'r-', label='Seq Acc %')
    ax2_twin.set_ylabel('Sequence Accuracy (%)', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('F1 Score vs Sequence Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    
    train_csv = 'train.csv'
    test_csv = 'test.csv'

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    vocab = Vocab()
    print(f'Vocabulary size: {vocab.num_tokens}')

    # Load datasets
    train_dataset = MazeDataset(train_csv, vocab)
    test_dataset = MazeDataset(test_csv, vocab)
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = RNNModel(
        input_dim=vocab.num_tokens,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # History dictionary to store all metrics
    history = {
        'train_loss': [],
        'train_token_acc': [],
        'train_f1': [],
        'test_loss': [],
        'test_token_acc': [],
        'test_seq_acc': [],
        'test_f1': []
    }
    
    best_test_seq_acc = 0.0

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 60)
        
        # Training
        train_loss, train_token_acc, train_f1 = train_for_one_epoch(
            model, train_loader, criterion, optimizer, device, tcher_force
        )
        
        # Testing
        test_loss, test_token_acc, test_seq_acc, test_f1 = evaluate(
            model, test_loader, criterion, device
        )
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_token_acc'].append(train_token_acc)
        history['train_f1'].append(train_f1)
        history['test_loss'].append(test_loss)
        history['test_token_acc'].append(test_token_acc)
        history['test_seq_acc'].append(test_seq_acc)
        history['test_f1'].append(test_f1)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f} | Train Token Acc: {train_token_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Test Loss: {test_loss:.4f} | Test Token Acc: {test_token_acc:.4f} | Test Seq Acc: {test_seq_acc:.4f} | Test F1: {test_f1:.4f}')
        
        # Save best model
        if test_seq_acc > best_test_seq_acc:
            best_test_seq_acc = test_seq_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_seq_acc': test_seq_acc,
                'vocab': vocab,
                'history': history
            }, 'best_rnn_model.pth')
            print(f'✓ Best model saved! (Test Seq Acc: {test_seq_acc:.4f})')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'vocab': vocab
            }, f'rnn_model_epoch_{epoch + 1}.pth')
            print(f'✓ Checkpoint saved at epoch {epoch + 1}')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'history': history
    }, 'rnn_model_final.pth')
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f'Best Test Sequence Accuracy: {best_test_seq_acc:.4f}')
    
    # Plot training curves
    plot_training_curves(history, 'rnn_training_curves.png')
    
    # Get random predictions from test set
    print("\n" + "="*60)
    print("Generating Random Predictions")
    print("="*60)
    
    random_predictions = get_random_predictions(model, test_dataset, vocab, device, num_samples=5)
    
    # Save predictions to JSON file
    with open('random_predictions.json', 'w') as f:
        json.dump(random_predictions, f, indent=2)
    
    print("✓ Random predictions saved to 'random_predictions.json'")
    
    # Also save as readable text file
    with open('random_predictions.txt', 'w') as f:
        for i, pred in enumerate(random_predictions, 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"Sample {i}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Input Sequence:\n{' '.join(pred['input_sequence'])}\n\n")
            f.write(f"Target Sequence:\n{' '.join(pred['target_sequence'])}\n\n")
            f.write(f"Predicted Sequence:\n{' '.join(pred['predicted_sequence'])}\n\n")
    
    print("✓ Random predictions saved to 'random_predictions.txt'")
    
    # Save history as CSV for easy analysis
    history_df = pd.DataFrame(history)
    history_df.index.name = 'epoch'
    history_df.to_csv('training_history.csv')
    print("✓ Training history saved to 'training_history.csv'")
    
    print("\n" + "="*60)
    print("All outputs saved successfully!")
    print("="*60)
