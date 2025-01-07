import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from collections import Counter
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)
processed_data = {'train': [], 'validation': [], 'test': []}

#Process each split and include start (<s>) and stop (</s>) tokens
for split in ['train', 'validation', 'test']:
    for i in range(len(ptb[split])):
        sentence = ptb[split][i]['sentence']
        tokens = '<s> ' + sentence + ' </s>'
        processed_data[split].append(tokens)

#Tokenize and build vocabulary on the training set only
tokenized_train = [sentence.split() for sentence in processed_data['train']]
vocab_counter = Counter(token for sentence in tokenized_train for token in sentence)

#Create token-to-index mapping
vocab = {token: idx for idx, (token, _) in enumerate(vocab_counter.most_common())}

#Map tokens to indices for all splits
mapped_data = {
    split: [[vocab[token] for token in sentence.split()] for sentence in processed_data[split]]
    for split in ['train', 'validation', 'test']
}

max_length = 50

#Process each split: truncate and pad
padded_data = {}
for split in ['train', 'validation', 'test']:
    truncated_data = [sentence[:max_length] for sentence in mapped_data[split]]
    padded_data[split] = [sentence + [0] * (max_length - len(sentence)) for sentence in truncated_data]

#Convert each split to tensors
tensor_data = {split: torch.tensor(padded_data[split], dtype=torch.long) for split in ['train', 'validation', 'test']}

data_inputs = tensor_data['train'][:, :-1]  #All tokens except the last
data_targets = tensor_data['train'][:, 1:]  #All tokens except the first

#Split newly established inputs and targets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(data_inputs, data_targets, test_size=0.2, random_state=42)

#Initialize training and testing datasets and loaders
train_data = TensorDataset(train_inputs, train_targets)
val_data = TensorDataset(val_inputs, val_targets)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

#sin and cos positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

#Transformer architecture
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)

        self.fc = nn.Linear(d_model, vocab_size)

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, x):  # x: (batch, seq_len)
      x = self.embedding(x)  # (batch, seq_len, d_model)
      x = self.pos_encoding(x)  # (batch, seq_len, d_model)
      x = self.dropout(x)

      # Generate a causal mask for the sequence
      seq_len = x.size(1)
      mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)  # (seq_len, seq_len)

     # Transpose x for TransformerEncoder compatibility
      x = self.transformer_encoder(x.transpose(0, 1), mask=mask)  # (seq_len, batch, d_model)

      # Transpose back
      x = x.transpose(0, 1)  # (batch, seq_len, d_model)

      # Apply the linear layer to get logits
      out = self.fc(x)  # (batch, seq_len, vocab_size)
      return out

vocab_size = len(vocab)
d_model = 256
n_head = 4
n_layer = 4
batch_size = 32
learning_rate = 0.0005388906778411895
num_epochs = 20
dropout = 0.2

model = TransformerLM(vocab_size, d_model, n_head, n_layer)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

model = model.to(device)
criterion = criterion.to(device)

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        #Forward pass
        outputs = model(inputs)  

        #Reshape for loss computation
        logits = outputs.view(-1, vocab_size)
        targets = targets.view(-1)            

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  #Store train loss
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    #Validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            val_outputs = model(val_inputs)

            val_logits = val_outputs.view(-1, vocab_size)
            val_targets = val_targets.view(-1)
            val_loss += criterion(val_logits, val_targets).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # Store validation loss
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')

# Customize the plot
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

test_inputs = tensor_data['test'][:, :-1]  #All tokens except the last
test_targets = tensor_data['test'][:, 1:]  #All tokens except the first

#Create a dataset combining inputs and targets for test
test_dataset = TensorDataset(test_inputs, test_targets)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = model.to(device)

def calculate_test_set_perplexities(model, test_loader):
    model.eval()
    results = []  #To store perplexities for each sequence
    total_perplexity = 0  #To accumulate perplexities
    total_sequences = 0  #To count sequences
    id_counter = 0  #ID counter for CSV format

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)  #Input sequences
            targets = batch[1].to(device)  #Target sequences

            #Get model predictions
            logits = model(inputs) 
            probs = F.softmax(logits, dim=-1)  #Convert logits to probabilities

            #Gather probabilities for target tokens
            token_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            #Compute log probabilities
            log_probs = torch.log(token_probs)
            sequence_log_probs = log_probs.sum(dim=1)  #Sum log probs for each sequence
            sequence_lengths = (targets != 0).sum(dim=1)  #Count non-padding tokens

            #Compute perplexity for each sequence in the batch
            perplexities = [
                math.exp(-log_prob / length.item())
                for log_prob, length in zip(sequence_log_probs, sequence_lengths)
            ]

            #Store results with IDs
            for perplexity in perplexities:
                results.append({'ID': id_counter, 'ppl': perplexity})
                total_perplexity += perplexity  #Accumulate perplexity
                total_sequences += 1  #Count the sequence
                id_counter += 1

    #Compute average perplexity
    average_perplexity = total_perplexity / total_sequences

    return results, average_perplexity

#Compute perplexities and average perplexity
test_results, avg_perplexity = calculate_test_set_perplexities(model, test_loader)

#Save results to a CSV file
df = pd.DataFrame(test_results)
df.to_csv('submission.csv', index=False)

print(f"Perplexities saved to 'submission.csv'")
print(f"Average Perplexity: {avg_perplexity}")
