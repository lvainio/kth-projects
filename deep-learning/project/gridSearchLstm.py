import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from data import Data


class LSTM(nn.Module):

    def __init__(self, num_features, hidden_units, embedding_dim, num_layers, temperature, data, p, dropout, device):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.temperature = temperature
        self.data = data
        self.p = p
        self.lstm = nn.LSTM( 
            input_size=embedding_dim,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_units, num_features)
        self.h = nn.Parameter(torch.zeros(num_layers, 1, hidden_units))
        self.c = nn.Parameter(torch.zeros(num_layers, 1, hidden_units))
        self.device = device
        
    def forward(self, x):
        embedded = self.embedding(x)
        h_expanded = self.h.expand(-1, x.size(0), -1).contiguous()
        c_expanded = self.c.expand(-1, x.size(0), -1).contiguous()
        y, _ = self.lstm(embedded, (h_expanded, c_expanded))
        y = self.fc(y)
        return y
    
    def nucleus_sampling(self, probs):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumulative_probs < self.p 
        n = torch.sum(mask).item() + 1
        top_p = sorted_probs[:n]
        top_p_probs = top_p / torch.sum(top_p)
        top_idx = torch.multinomial(top_p_probs, 1)
        return sorted_indices[top_idx] 
        
    def synthesize(self, n):
        x = torch.tensor(0).view(1, 1).to(self.device)
        h_prev = self.h.clone()
        c_prev = self.c.clone()
        chars = []
        with torch.no_grad():
            for _ in range(n):
                emb = self.embedding(x)
                out, (h_prev, c_prev) = self.lstm(emb, (h_prev, c_prev))
                logits = self.fc(out).view(-1) / self.temperature
                probs = F.softmax(logits, dim=0)
                # x = torch.multinomial(probs, 1).view(1, 1)
                x = self.nucleus_sampling(probs).view(1, 1)
                chars.append(self.data.index_to_char[x.item()])
        return ''.join(chars)


def train_and_evaluate(lstm, data, optimizer, loss_fn, device, epochs, num_train_batches, num_val_batches):
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, epochs+1):
        lstm.train()
        for i, (inputs, outputs) in enumerate(data.train_dataloader):
            inputs, outputs = inputs.to(device), outputs.to(device)
            optimizer.zero_grad()
            logits = lstm.forward(inputs).permute(0, 2, 1)
            one_hot_outputs = torch.nn.functional.one_hot(outputs, num_classes=lstm.num_features) 
            loss = loss_fn(logits, one_hot_outputs.to(torch.float32).permute(0, 2, 1))
            loss.backward()
            optimizer.step()
        
        # Validation
        lstm.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, outputs in data.val_dataloader:
                logits = lstm.forward(inputs.to(device)).permute(0, 2, 1)
                one_hot_outputs = torch.nn.functional.one_hot(outputs, num_classes=lstm.num_features).to(device)
                val_loss += loss_fn(logits, one_hot_outputs.to(torch.float32).permute(0, 2, 1)).item()
        val_loss = val_loss / num_val_batches
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def main():
    hidden_units = 1024
    sequence_length = 100
    epochs = 15
    temperature = 0.85
    train_split = 0.7
    val_split = 0.1
    test_split = 0.2
    p = 0.95
    weight_decay = 0
    embedding_dim = 256
    num_layers = 2
    dropout = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_sizes = [32, 64, 128]
    learning_rates = [0.0001, 0.0005, 0.001]

    best_params = None
    best_val_loss = float('inf')

    for batch_size, learning_rate in itertools.product(batch_sizes, learning_rates):
        print(f"\n##### Training with batch_size={batch_size}, learning_rate={learning_rate} #####")
        
        data = Data(sequence_length, batch_size, train_split, val_split, test_split)
        num_train_batches = len(data.train_dataloader)
        num_val_batches = len(data.val_dataloader)
        num_test_batches = len(data.test_dataloader)
        num_features = data.vocabulary_size

        lstm = LSTM(num_features, hidden_units, embedding_dim, num_layers, temperature, data, p, dropout, device).to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=weight_decay)

        val_loss = train_and_evaluate(lstm, data, optimizer, loss_fn, device, epochs, num_train_batches, num_val_batches)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (batch_size, learning_rate)
        
        print(f"Validation Loss: {val_loss}")

    print(f"\nBest Parameters: Batch Size={best_params[0]}, Learning Rate={best_params[1]}")
    print(f"Best Validation Loss: {best_val_loss}")

    # Final training with best parameters
    print("\n##### Training with best parameters #####")
    batch_size, learning_rate = best_params

    data = Data(sequence_length, batch_size, train_split, val_split, test_split)
    num_train_batches = len(data.train_dataloader)
    num_val_batches = len(data.val_dataloader)
    num_test_batches = len(data.test_dataloader)
    num_features = data.vocabulary_size

    lstm = LSTM(num_features, hidden_units, embedding_dim, num_layers, temperature, data, p, dropout, device).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_and_evaluate(lstm, data, optimizer, loss_fn, device, epochs, num_train_batches, num_val_batches)
    
    # Final results
    lstm.eval()
    train_loss = 0
    with torch.no_grad():
        for inputs, outputs in data.train_dataloader:
            logits = lstm.forward(inputs.to(device)).permute(0, 2, 1)
            one_hot_outputs = torch.nn.functional.one_hot(outputs, num_classes=num_features).to(device)
            train_loss += loss_fn(logits, one_hot_outputs.to(torch.float32).permute(0, 2, 1)).item()
    train_loss = train_loss / num_train_batches
    val_loss = 0
    with torch.no_grad():
        for inputs, outputs in data.val_dataloader:
            logits = lstm.forward(inputs.to(device)).permute(0, 2, 1)
            one_hot_outputs = torch.nn.functional.one_hot(outputs, num_classes=num_features).to(device)
            val_loss += loss_fn(logits, one_hot_outputs.to(torch.float32).permute(0, 2, 1)).item()
    val_loss = val_loss / num_val_batches
    test_loss = 0
    with torch.no_grad():
        for inputs, outputs in data.test_dataloader:
            logits = lstm.forward(inputs.to(device)).permute(0, 2, 1)
            one_hot_outputs = torch.nn.functional.one_hot(outputs, num_classes=num_features).to(device) 
            test_loss += loss_fn(logits, one_hot_outputs.to(torch.float32).permute(0, 2, 1)).item()
    test_loss = test_loss / num_test_batches
    print(f"Final results:")
    print(f" - Train loss: {train_loss}")
    print(f" - Val loss: {val_loss}")
    print(f" - Test loss: {test_loss}") 
    text = lstm.synthesize(2000)
    print(f"Synthesized text (2000 characters): \n{text}")

if __name__ == "__main__":
    main()
