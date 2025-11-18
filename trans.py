#transformer architecture
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ast
import re

D_model = 128
nhead = 8
num_layers = 6
dim_ff = 512
dropout = 0.1
pad_index = 0
EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

class transformer(nn.Module):
    def __init__(self,input_dim,D_model,nhead,num_layers,dim_ff,dropout,pad_index,max_len=5000):
        super(transformer,self).__init__()
        self.d_model=D_model
        self.pad_index=pad_index
        
        self.embedlayer=nn.Embedding(input_dim,D_model,padding_idx=pad_index)
        pe = self.positional_encoding(max_len,D_model)
        self.register_buffer('pos_encode', pe)
        self.dropout=nn.Dropout(dropout)
        encode_layer=nn.TransformerEncoderLayer(d_model=D_model,nhead=nhead,dim_feedforward=dim_ff,dropout=dropout,batch_first=True)
        self.encoder=nn.TransformerEncoder(encode_layer,num_layers=num_layers)
        decode_layer=nn.TransformerDecoderLayer(d_model=D_model,nhead=nhead,dim_feedforward=dim_ff,dropout=dropout,batch_first=True)
        self.decoder=nn.TransformerDecoder(decode_layer,num_layers=num_layers)
        self.out_layer=nn.Linear(D_model,input_dim)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def positional_encoding(self,max_len,D_model):
        pos_enc=torch.zeros(max_len,D_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_model, 2).float() * (-math.log(10000.0) / D_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc=pos_enc.unsqueeze(0)
        return pos_enc
    
    def add_positional_encoding(self,input):
        input=input+self.pos_encode[:, :input.size(1), :].to(input.device)
        return self.dropout(input)
        
    def create_mask(self, src, tgt):
        tgt_seq_len = tgt.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        src_padmask = (src == self.pad_index)
        tgt_padmask = (tgt == self.pad_index)
        return tgt_mask, src_padmask, tgt_padmask
    
    def encode(self,src,src_mask):
        src_embed=self.embedlayer(src)*math.sqrt(self.d_model)
        src_embed=self.add_positional_encoding(src_embed)
        enc_out=self.encoder(src_embed,src_key_padding_mask=src_mask)
        return enc_out
    
    def decode(self,tgt,enc_out,tgt_mask,src_mask,tgt_padmask):
        tgt_embed=self.embedlayer(tgt)*math.sqrt(self.d_model)
        tgt_embed=self.add_positional_encoding(tgt_embed)
        dec_out=self.decoder(tgt_embed,enc_out,tgt_mask=tgt_mask,
                             memory_key_padding_mask=src_mask,
                             tgt_key_padding_mask=tgt_padmask)
        dec_out=self.out_layer(dec_out)
        return dec_out
    
    def forward(self,src,tgt):
        tgt_mask, src_padmask, tgt_padmask = self.create_mask(src, tgt)
        enc_out = self.encode(src, src_padmask)
        output = self.decode(tgt, enc_out, tgt_mask, src_padmask, tgt_padmask)
        return output
    
    def predict(self,src,max_len,start_token,end_token):
        self.eval()
        src_padmask = (src == self.pad_index)
        enc_out = self.encode(src, src_padmask)
        tgt_indexes = torch.full((src.size(0),1), start_token, dtype=torch.long, device=src.device)
        
        for i in range(max_len-1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_indexes.size(1)).to(src.device)
            tgt_padmask = (tgt_indexes == self.pad_index)
            out = self.decode(tgt_indexes, enc_out, tgt_mask, src_padmask, tgt_padmask)
            pred = out[:,-1,:].argmax(1).unsqueeze(1)
            tgt_indexes = torch.cat((tgt_indexes, pred), dim=1)
            if (pred == end_token).all():
                break
        return tgt_indexes





#visuaisation
def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    """Accepts many tag styles: <TAG_START>, <TAG START>, <TAG-START>, <TAGSTART>, etc."""
    patterns = [
        rf"<\s*{tag}\s*[_\-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[_\-\s]?\s*END\s*>",
        rf"<\s*{tag}START\s*>(.*?)<\s*{tag}END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
        rf"<\s*{tag.replace(' ', '_')}\s*START\s*>(.*?)<\s*{tag.replace(' ', '_')}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    return "" # Return empty string instead of raising error for robustness

def plot_maze(tokens, title="Maze"):
    text = " ".join(tokens)
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)
    path_section = extract_between("PATH", text)


    origin = parse_coords(origin_section)
    target = parse_coords(target_section)

    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        if len(coords) >= 2:
            a = parse_coords(coords[0])
            b = parse_coords(coords[1])
            edges.append((a, b))

    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    
    rows = 6
    cols = 6

    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            c_between = min(c1, c2) + 1
            if c_between < cols + 1: vertical_walls[r1, c_between] = False
        elif c1 == c2:
            r_between = min(r1, r2) + 1
            if r_between < rows + 1: horizontal_walls[r_between, c1] = False

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.set_title(title)

    # Draw grid
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=2)
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=2)
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=2)
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=2)

    # Draw walls
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=5, solid_capstyle='butt')

    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=5, solid_capstyle='butt')

    # Draw Path
    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        if len(path_x) > 0:
            ax.scatter(path_x[0], path_y[0], c='red', s=80, marker='o', zorder=5)
            ax.scatter(path_x[-1], path_y[-1], c='red', s=80, marker='x', zorder=5)
    
    # Draw Origin/Target if path not full or just to be safe
    if origin:
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        ax.scatter(ox, oy, c='green', s=80, marker='o', zorder=5, label='Origin')
    if target:
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.scatter(tx, ty, c='blue', s=80, marker='x', zorder=5, label='Target')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    plt.axis('off')
    plt.show()




def build_vocab(df):
    vocab = {'<PAD>', '<PATH_START>', '<PATH_END>', '<ADJLIST_START>', 
             '<ADJLIST_END>', '<ORIGIN_START>', '<ORIGIN_END>', 
             '<TARGET_START>', '<TARGET_END>', '<-->', ';', ','}
    for r in range(6):
        for c in range(6):
            vocab.add(f"({r},{c})")
    token_to_idx = {token: idx for idx, token in enumerate(sorted(list(vocab)))}
    if '<PAD>' in token_to_idx:
        pad_idx_old = token_to_idx['<PAD>']
        token_0 = [k for k, v in token_to_idx.items() if v == 0][0]
        token_to_idx['<PAD>'] = 0
        token_to_idx[token_0] = pad_idx_old
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    return token_to_idx, idx_to_token

def prepare_data(df, token_to_idx):
    all_src = []
    all_tgt = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        input_seq = ast.literal_eval(row['input_sequence'])
        output_seq = ast.literal_eval(row['output_path'])
        src_indices = [token_to_idx.get(t, 0) for t in input_seq] # safe get
        tgt_indices = [token_to_idx.get(t, 0) for t in output_seq]
        all_src.append(torch.tensor(src_indices))
        all_tgt.append(torch.tensor(tgt_indices))
    return all_src, all_tgt

def create_batches(src_list, tgt_list, batch_size):
    num_samples = len(src_list)
    indices = torch.randperm(num_samples).tolist()
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        src_batch = [src_list[j] for j in batch_indices]
        tgt_batch = [tgt_list[j] for j in batch_indices]
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
        batches.append((src_padded, tgt_padded))
    return batches

def train_epoch(model, src_list, tgt_list, batch_size, optimizer, criterion, device):
    model.train()
    batches = create_batches(src_list, tgt_list, batch_size)
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    
    for src, tgt in batches:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        output_flat = output.reshape(-1, output.shape[-1])
        tgt_output_flat = tgt_output.reshape(-1)
        
        loss = criterion(output_flat, tgt_output_flat)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = output.argmax(dim=-1)
        mask = (tgt_output != 0)
        correct_tokens += ((preds == tgt_output) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        for b in range(preds.shape[0]):
            pred_seq = preds[b][mask[b]]
            tgt_seq = tgt_output[b][mask[b]]
            if torch.equal(pred_seq, tgt_seq):
                correct_sequences += 1
            total_sequences += 1
        
    return total_loss / len(batches), correct_tokens / total_tokens, correct_sequences / total_sequences

def evaluate(model, src_list, tgt_list, batch_size, criterion, device):
    model.eval()
    batches = []
    for i in range(0, len(src_list), batch_size):
        src_batch = src_list[i:i+batch_size]
        tgt_batch = tgt_list[i:i+batch_size]
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
        batches.append((src_padded, tgt_padded))
    
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    
    with torch.no_grad():
        for src, tgt in batches:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_output_flat = tgt_output.reshape(-1)
            loss = criterion(output_flat, tgt_output_flat)
            total_loss += loss.item()
            
            preds = output.argmax(dim=-1)
            mask = (tgt_output != 0)
            correct_tokens += ((preds == tgt_output) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            for b in range(preds.shape[0]):
                pred_seq = preds[b][mask[b]]
                tgt_seq = tgt_output[b][mask[b]]
                if torch.equal(pred_seq, tgt_seq):
                    correct_sequences += 1
                total_sequences += 1
            
    return total_loss / len(batches), correct_tokens / total_tokens, correct_sequences / total_sequences

def compute_f1_score(pred_tokens, gt_tokens):
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    
    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0
    
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.plot(epochs, history['test_loss'], 'g--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_token_acc'], 'b-', label='Train Token Acc')
    plt.plot(epochs, history['val_token_acc'], 'r-', label='Val Token Acc')
    plt.plot(epochs, history['test_token_acc'], 'g--', label='Test Token Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Token Accuracy')
    plt.title('Token Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_seq_acc'], 'b-', label='Train Seq Acc')
    plt.plot(epochs, history['val_seq_acc'], 'r-', label='Val Seq Acc')
    plt.plot(epochs, history['test_seq_acc'], 'g--', label='Test Seq Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Sequence Accuracy')
    plt.title('Sequence Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer_metrics.png')
    plt.show()

def visualize_predictions(model, src_list, tgt_list, idx_to_token, token_to_idx, device, num_samples=5):
    model.eval()
    indices = np.random.choice(len(src_list), min(num_samples, len(src_list)), replace=False)
    
    start_token = token_to_idx['<PATH_START>']
    end_token = token_to_idx['<PATH_END>']
    
    print("\n=== Visualizing Predictions ===")
    for idx in indices:
        src = src_list[idx].unsqueeze(0).to(device)
        tgt = tgt_list[idx]
        
        with torch.no_grad():
            pred_seq = model.predict(src, max_len=100, start_token=start_token, end_token=end_token)
        
        src_tokens = [idx_to_token[i.item()] for i in src_list[idx] if i.item() != 0]
        tgt_tokens = [idx_to_token[i.item()] for i in tgt if i.item() != 0]
        pred_tokens = [idx_to_token[i.item()] for i in pred_seq[0] if i.item() != 0]
        
        f1 = compute_f1_score(pred_tokens, tgt_tokens)
        exact_match = (pred_tokens == tgt_tokens)
        
        print(f"\nSample {idx}:")
        print(f"Ground Truth: {' '.join(tgt_tokens)}")
        print(f"Predicted:    {' '.join(pred_tokens)}")
        print(f"Exact Match: {exact_match} | F1 Score: {f1:.4f}")

        plot_maze(src_tokens + pred_tokens, title=f"Prediction Sample {idx}")

if __name__ == "__main__":
    
    train_full_df = pd.read_csv('train_6x6_mazes.csv')
    test_df = pd.read_csv('test_6x6_mazes.csv')
    
    train_df, val_df = train_test_split(train_full_df, test_size=0.1, random_state=42)


    # 2. Build Vocabulary 
    token_to_idx, idx_to_token = build_vocab(train_full_df)
    vocab_size = len(token_to_idx)
    print(f"Vocabulary Size: {vocab_size}")
    
    # 3. Prepare Tensors
    train_src, train_tgt = prepare_data(train_df, token_to_idx)
    val_src, val_tgt = prepare_data(val_df, token_to_idx)
    test_src, test_tgt = prepare_data(test_df, token_to_idx)

    # 4. Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = transformer(vocab_size, D_model, nhead, num_layers, dim_ff, dropout, pad_index).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
  
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_token_acc': [], 'val_token_acc': [], 'test_token_acc': [],
        'train_seq_acc': [], 'val_seq_acc': [], 'test_seq_acc': []
    }

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # Train
        t_loss, t_token_acc, t_seq_acc = train_epoch(model, train_src, train_tgt, BATCH_SIZE, optimizer, criterion, device)
        
        # Validate
        v_loss, v_token_acc, v_seq_acc = evaluate(model, val_src, val_tgt, BATCH_SIZE, criterion, device)
        
        # Test
        test_loss, test_token_acc, test_seq_acc = evaluate(model, test_src, test_tgt, BATCH_SIZE, criterion, device)
        
        # Store
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['test_loss'].append(test_loss)
        
        history['train_token_acc'].append(t_token_acc)
        history['val_token_acc'].append(v_token_acc)
        history['test_token_acc'].append(test_token_acc)

        history['train_seq_acc'].append(t_seq_acc)
        history['val_seq_acc'].append(v_seq_acc)
        history['test_seq_acc'].append(test_seq_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train - Loss: {t_loss:.4f} | Token Acc: {t_token_acc:.4f} | Seq Acc: {t_seq_acc:.4f}")
        print(f"  Val   - Loss: {v_loss:.4f} | Token Acc: {v_token_acc:.4f} | Seq Acc: {v_seq_acc:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f} | Token Acc: {test_token_acc:.4f} | Seq Acc: {test_seq_acc:.4f}")

    # Plot metrics
    plot_metrics(history)
    
    # Visualize on Validation Set
    visualize_predictions(model, val_src, val_tgt, idx_to_token, token_to_idx, device, num_samples=5)
    
    # 7. Save Model
    save_data = {
        'model_state_dict': model.state_dict(),
        'vocab': token_to_idx,
        'hyperparameters': {
            'input_dim': vocab_size,
            'D_model': D_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_ff': dim_ff,
            'dropout': dropout,
            'pad_index': pad_index
        }
    }
    torch.save(save_data, 'transformer_model.pth')
    print("Model saved to 'transformer_model.pth'")
