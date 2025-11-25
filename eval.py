import torch 
import torch.nn as nn
import pandas as pd
import sys
import os

# Import rnn module
import rnn

# Import Vocab to __main__ namespace for pickle compatibility
# This is needed because checkpoints may have Vocab objects pickled as __main__.Vocab
from rnn import Vocab


def load_model(model_path, vocab_size, device, model_type='rnn'):
    """Load pretrained model"""
    if model_type == 'rnn':
        model = rnn.RNNModel(
            input_dim=vocab_size,
            embed_dim=rnn.embed_dim,
            hidden_dim=rnn.hidden_dim,
            num_layers=rnn.num_layers
        ).to(device)
    else:
        raise ValueError(f"Model type '{model_type}' not supported yet. Use 'rnn'.")
    
    # Load state dict with weights_only=False for backwards compatibility
    # This is safe if you trust the checkpoint file source
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_sequence(model, input_seq, vocab, device, max_length=100):
    """Generate prediction for a single input sequence"""
    model.eval()
    
    with torch.no_grad():
        # Encode input sequence
        input_encoding = vocab.encode_sequence(input_seq)
        input_tensor = torch.tensor([input_encoding]).to(device)
        input_length = torch.tensor([len(input_encoding)]).to(device)
        
        # Encode input
        encoder_outputs, hidden = model.encode(input_tensor, input_length)
        
        # Create mask
        mask = (input_tensor != rnn.pad_index).to(device)
        
        # Initialize decoder with <PATH_START> token (index 7)
        decoder_input = torch.tensor([7]).to(device)
        decoder_hidden = hidden
        
        # Store predictions
        predictions = [7]  # Start with <PATH_START>
        
        start=""
        for index, i in enumerate(input_seq):
            if(i=="<ORIGIN_START>"):
                start=input_seq[index+1]
                break
        # Generate sequence
        for t in range(max_length):
            decoder_output, decoder_hidden = model.decode(
                decoder_input, encoder_outputs, decoder_hidden, mask
            )
            
            # Get most likely token
            top1 = decoder_output.argmax(1).item()
            predictions.append(top1)
            
            # Stop if <PATH_END> token (index 8) is generated
            if top1 == 8:
                break
            
            decoder_input = torch.tensor([top1]).to(device)
        
        # Decode predictions back to tokens
        predictions.pop(0)
        
        output_path = vocab.decode_sequence(predictions)
        output_path.insert(0, start)
        
        return output_path


def generate_predictions(model, input_csv, output_csv, vocab, device):
    """Generate predictions for all rows in input CSV"""
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    print(f"Processing {len(df)} rows...")
    
    # Store predictions
    output_paths = []
    
    # Process each row
    for idx in range(len(df)):
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows...")
        
        row = df.iloc[idx]
        
        # Parse input sequence
        input_seq = eval(row['input_sequence'])
        
        # Generate prediction
        output_path = predict_sequence(model, input_seq, vocab, device)
        
        # Convert output path list to string representation
        output_paths.append(str(output_path))
    
    # Add output_path column to dataframe
    df['output_path'] = output_paths
    
    # Save to output CSV
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 eval.py <path_to_pretrained_model> <type_of_model> <path_to_input.csv> <output.csv>")
        print("Example: python3 eval.py rnn_model_final.pth rnn test.csv output.csv")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    input_csv = sys.argv[3]
    output_csv = sys.argv[4]
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        sys.exit(1)
    
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file '{input_csv}' not found")
        sys.exit(1)
    
    if model_type.lower() not in ['rnn', 'transformer']:
        print(f"Error: Model type must be 'rnn' or 'transformer', got '{model_type}'")
        sys.exit(1)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize vocabulary
    if(model_type=='rnn'):
        vocab = rnn.Vocab()
        print(f"Vocabulary size: {vocab.num_tokens}")
        
        # Load model
        print(f"Loading {model_type.upper()} model from {model_path}...")
        model = load_model(model_path, vocab.num_tokens, device, model_type.lower())
        print("Model loaded successfully!")
        
        # Generate predictions
        generate_predictions(model, input_csv, output_csv, vocab, device)
        print("Done!")
    

if __name__ == '__main__':
    main()