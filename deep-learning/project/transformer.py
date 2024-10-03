import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split



batch_size = 64 
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
train_split = 0.8
val_split = 0.1
test_split = 0.1
epochs = 100



def download_text():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
    else:
        raise Exception(f"Could not download {url}")
    return text



text = download_text()
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
char_to_index = {char: index for index, char in enumerate(vocab)}
index_to_char = {index: char for index, char in enumerate(vocab)}
text_as_indices = torch.tensor([char_to_index[char] for char in text], dtype=torch.long)
num_samples = len(text_as_indices) // (block_size + 1) 
inputs = torch.reshape(text_as_indices[:num_samples*block_size], (-1, block_size))
outputs = torch.reshape(text_as_indices[1:1+num_samples*block_size], (-1, block_size))
dataset = TensorDataset(inputs, outputs)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_split, val_split, test_split])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)



def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(len(dataloader))
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            _, loss = model(src, trg)
            losses[i] = loss.item()
    return losses.mean()

def train(model, optimizer, dataloader):
    model.train()
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(src, trg)
        loss.backward()
        optimizer.step()

def generate(model, num_tokens):
    model.eval()
    indices = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(num_tokens):
            indices_block = indices[:, -block_size:]
            logits, _ = model(indices_block)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            indices = torch.cat((indices, idx_next), dim=1) 
    return "".join([index_to_char[i] for i in indices[0].tolist()])



class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss



model = TransformerModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    val_loss = evaluate(model, val_dataloader)
    print(f"epoch {epoch+1}: val loss {val_loss:.4f}")
    train(model, optimizer, train_dataloader)

train_loss = evaluate(model, train_dataloader)
val_loss = evaluate(model, val_dataloader)
test_loss = evaluate(model, test_dataloader)

print("\n\n\nFinal results:")
print(f" - Train loss: {train_loss}")
print(f" - Val loss: {val_loss}")
print(f" - Test loss: {test_loss}\n")
print("Synthesized text 3000 chars:")
print(generate(model, 3000))