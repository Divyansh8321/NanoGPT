#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F

# reading the dataset
with open("input.txt" , 'r' , encoding = 'utf-8') as f:
    words = f.read()   

# Generating the shakespearean vocabulary
chars = sorted(list(set(words)))
vocab_size = len(chars)

# Creating the stoi and itos mappings
stoi = {s:i for i , s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}

# encoding and decoding strings 
encode = lambda s : [stoi[ch] for ch in s]
decode = lambda l : ''.join([itos[ch] for ch in l])

# Encoding the entire tiny shakespeare dataset into integers and wrap it in a torch tensor
data = torch.tensor(encode(words) , dtype = torch.long)

# Creating train and cross-val sets
n1 = int(len(data)* 0.9)
train_data = data[:n1]
cv_data = data[:n1]

torch.manual_seed(47)
# hyperparameters
block_size = 64 # This is the length of the chunk trained on ; can also be called context_length
batch_size = 256 # This is the number of such chunks trained on in parallel
max_iters = 5000 # This is the number of steps after which the training is stopped
lr = 1e-4 # This is the learning rate of the model
eval_iters = 100 # This is the number of steps for which the model is evaluated on the cross-val set
eval_interval = 500 # This is the interval after which the model is evaluated on the cross-val set
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embd_size = 384 # This is the size of the embedding vector
head_size = 6 # This is the size of the attention head
n_layers = 6 # This is the number of layers in the transformer
dropout = 0.2 # This is the dropout rate


# This function will return a random batch of data of size batch_size and block_size
def get_batch(split):
    data = train_data if split == 'train' else cv_data
    ix = torch.randint(len(data) - batch_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y

# Evaluating the val and the train loss in intervals of eval_iters so that the loss curve is smooth 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'cv']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb , yb = get_batch(split)
            logits , loss = model(xb,yb)
            losses[k] = loss.item()
        out[split + '_loss'] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    """One head of the multi-head attention block"""

    def __init__(self , head_size):
        super().__init__()
        self.query = nn.Linear(embd_size , head_size , bias = False)
        self.key = nn.Linear(embd_size , head_size , bias = False)
        self.value = nn.Linear(embd_size , head_size,  bias = False)
        self.register_buffer('tril' , torch.tril(torch.ones((block_size , block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        
        # q and  k  are B,T,head_size
        # compute the scaled dot product attention
        wei = q @ k.transpose(-2,-1) * (head_size)** (-0.5) # (B, T, head_size) @ (B, head_size, T) --> (B, T , T)
        # mask the upper triangular part of the matrix
        wei = wei.masked_fill(self.tril[:T,:T]==0 , float('-inf'))
        # apply softmax to get the affinity matrix
        wei = F.softmax(wei , dim = 1)
        # Aggregate the values from every character in the sequence
        v = self.value(x)
        # Adding dropout 
        wei = self.dropout(wei)

        out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out 


class MultiHeadAttention(nn.Module):
    def __init__(self , num_heads , head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embd_size , embd_size) # This is the projection layer which will project the output of the heads back to the original dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is B, T, C
        # concatenate the output of each head
        out = torch.cat([head(x) for head in self.heads] , dim = -1)
        out = self.dropout((self.proj(out)))
        return out 

class FeedForward(nn.Module):
    """A linear layer followed by a non-linear activation"""
    # This is on a per token level and all the tokens do this independantly
    def __init__(self , embd_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size ,4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size , embd_size), # Projecting back to the original dimension
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block : Communication followed by computatiton"""
    def __init__(self, embd_size , num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads , embd_size//num_heads)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x)) # The x = x + represents the skip connection which allows the gradient to flow unimpeded initially 
        x = x + self.ffwd(self.ln2(x))
        return x

# We will start off with the simplest model which is Bigram Language Model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size , embd_size)
        self.position_embedding_table = nn.Embedding(block_size , embd_size)
        self.blocks = nn.Sequential(
            *[Block(embd_size , head_size = head_size) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(embd_size , vocab_size)

    def forward(self, idx , targets = None):
        
        B,T = idx.shape
        # idx and targets bring B, T and the embedding table brings the dimension C
        tok_emb = self.token_embedding_table(idx) # B, T, embd_size
        pos_emb = self.position_embedding_table(torch.arange(T , device = device)) # T, embd_size
        x = tok_emb + pos_emb # B, T, embd_size
        x = self.saheads(x) # B, T, embd_size
        x = self.ffwd(x) # B, T, embd_size
        logits = self.lm_head(x) # B, T, C
        if targets is None :
            loss = None 
        else :
            B,T,C = logits.shape  
            # loss = F.cross_entropy(logits, targets)  This won't work because pytorch expects logits in the form B,C,T                 
            loss = F.cross_entropy(logits.view(B*T,C) , targets.view(B*T))    
            # Now channel dimensional is the second dimension as expected and targets becomes B*T instead of B,T 

        return logits , loss  

    def generate(self, idx, max_tokens): # This will sort of do the next word prediction till it generates
    # max_tokens 
        for _ in range(max_tokens):
            # cropping idx to ensure that the context does not exceed the block_size 
            idx_cond = idx[:,-block_size:]
            # idx is B, T
            logits , loss = self(idx_cond)
            # focusing only on the last time step 
            logits = logits[: , -1 , :] # becomes B, C 
            # applyig softmax 
            probs = F.softmax(logits , dim = -1) # also B, C
            # predict the next character in the sequence 
            idx_next = torch.multinomial(probs , num_samples = 1) # B, 1
            # append the predicted the character to the current sequence 
            idx = torch.cat((idx , idx_next), dim = 1) # B, T+1
        return idx 
        
model = BigramLanguageModel()
model = model.to(device)

# create the learning rate optimizer 
optimizer = torch.optim.AdamW(model.parameters() , lr = lr)

# Gradient descent 
for iter in range(max_iters):
    
    if iter%eval_interval == 0 :
        losses = estimate_loss()
        print (f'step {iter} : train_loss = {losses["train_loss"]:.4f} , cv_loss = {losses["cv_loss"]:.4f}')
    
    # sample a batch of data
    xb , yb = get_batch('train')

    # evaluate the loss
    logits , loss = model(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    
# Sampling from the model
context = torch.zeros((1,1) , dtype = torch.long , device = device)
print (decode(model.generate(context , max_tokens = 300)[0].tolist()))