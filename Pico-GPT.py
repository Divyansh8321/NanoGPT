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
block_size = 8 # This is the length of the chunk trained on ; can also be called context_length
batch_size = 4 # This is the number of such chunks trained on in parallel
max_iters = 10000 # This is the number of steps after which the training is stopped
lr = 1e-3 # This is the learning rate of the model
eval_iters = 100 # This is the number of steps for which the model is evaluated on the cross-val set
eval_interval = 100 # This is the interval after which the model is evaluated on the cross-val set
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embd_size = 32 # This is the size of the embedding vector


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


# We will start off with the simplest model which is Bigram Language Model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size , embd_size)
        self.position_embedding_table = nn.Embedding(block_size , embd_size)
        self.lm_head = nn.Linear(embd_size , vocab_size)

    def forward(self, idx , targets = None):
        
        # idx and targets bring B, T and the embedding table brings the dimension C
        tok_emb = self.token_embedding_table(idx) # B, T, embd_size
        pos_emb = self.position_embedding_table(torch.arange(block_size , device = device)) # T, embd_size
        final_emb = tok_emb + pos_emb # B, T, embd_size
        logits = self.lm_head(final_emb)
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
            # idx is B, T
            logits , _ = self(idx)
            # focusing only on the last time step 
            logits = logits[: , -1 , :] # becomes B, C 
            # applyig softmax 
            probs = F.softmax(logits , dim = 1) # also B, C
            # predict the next character in the sequence 
            idx_next = torch.multinomial(probs , num_samples = 1) # B, 1
            # append the predicted the character to the current sequence 
            idx = torch.cat((idx , idx_next), dim = 1) # B, T+1
        return idx 
        
model = BigramLanguageModel()
model = model.to(device)

# create the learning rate optimizer 
optimizer = torch.optim.AdamW(model.parameters() , lr = 1e-3)

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