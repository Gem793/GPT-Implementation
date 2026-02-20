import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size=64 #no of sequences gpu processes paralelly
block_size=256 #size of each individual sequence
max_iters=5000
eval_interval=500
learning_rate=3e-4
device="cuda" if torch.cuda.is_available() else "cpu"
eval_iters=200
n_enbd=384
n_head=6
n_layer=6
dropout=0.2

torch.manual_seed(1337)


with open('input.txt','r',encoding='utf-8') as f:
  text=f.read()


chars=sorted(list(set(text))) #set: unique items
vocab_size=len(chars)
#decoding and encoding functions
stoi={c:i for i,c in enumerate(chars)}
itos={i:c for i,c in enumerate(chars)}
encode=lambda s: [stoi[c] for c in s]
decode=lambda l: ''.join([itos[n] for n in l])

# train test data splitting
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[0:n]
val_data=data[n:]


#data loading
def get_batch(split):
  data=train_data if split=="train" else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))
  #if say len(Data)=100, starting value of batch seq can be max 91; len(data)-block_size=92, so [0:91] can be start values
  #(batch_size,): size argument; creates 1d tensor with 4 random start indices
  x=torch.stack([data[i:i+block_size] for i in ix])
  y=torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device) # Move tensors to the specified device
  return x,y

@torch.no_grad()
def estimate_loss():
  out={}
  model.eval()
  for split in ['train','val']:
    losses=torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y=get_batch(split)
      logits,loss=model(X,Y)
      losses[k]=loss.item()
    out[split]=losses.mean()
  model.train()
  return out

#one head of self attention
class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key=nn.Linear(n_enbd,head_size,bias=False)
    self.query=nn.Linear(n_enbd,head_size,bias=False)
    self.value=nn.Linear(n_enbd,head_size,bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    B,T,C=x.shape
    k=self.key(x)
    q=self.query(x)
      #computing attention scores
    wei=q@k.transpose(-2,-1)* C**-0.5
    wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei=F.softmax(wei,dim=-1)
    wei=self.dropout(wei)
    v=self.value(x)
    out=wei@v
    return out

#multi head attention
class MultiHeadAttention(nn.Module):
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj=nn.Linear(n_enbd,n_enbd)


  def forward(self,x):
    out=torch.cat([h(x) for h in self.heads],dim=-1)
    out=self.proj(out)
    return out

class FeedForward(nn.Module):
  def __init__(self,n_enbd):
    super().__init__()
    self.net=nn.Sequential(nn.Linear(n_enbd,4*n_enbd),
                           nn.ReLU(),
                           nn.Linear(4*n_enbd,n_enbd),
                           nn.Dropout(dropout),)

  def forward(self,x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self,n_enbd,n_head):
    super().__init__()
    head_size=n_enbd//n_head
    self.sa=MultiHeadAttention(n_head,head_size)
    self.ffwd=FeedForward(n_enbd)
    self.ln1=nn.LayerNorm(n_enbd)
    self.ln2=nn.LayerNorm(n_enbd)

  def forward(self,x):
    x=x+self.ln1(self.sa(x))
    x=x+self.ln2(self.ffwd(x))
    return x

#building the transformer
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.token_embedding_table=nn.Embedding(vocab_size, n_enbd)
    self.position_embedding_Table=nn.Embedding(block_size,n_enbd)
    self.blocks=nn.Sequential(*[Block(n_enbd,n_head=n_head) for _ in range(n_layer)])
    self.ln_f=nn.LayerNorm(n_enbd)
    self.lm_head=nn.Linear(n_enbd,vocab_size)

  def forward(self,idx,targets=None):
    B,T=idx.shape
    token_enbd=self.token_embedding_table(idx)
    pos_enbd=self.position_embedding_Table(torch.arange(T,device=device))
    x=token_enbd+pos_enbd
    x=self.blocks(x)
    logits=self.lm_head(x)#whatever tokens r in xb, we pull those vectors from embedding table and save them in logits[also a tensor]

    if targets is None:                        #comparing targets(yb) and the prob values(from logits), so if there r no targets, we can't compute loss
      loss=None
    else:
      B,T,C=logits.shape                       #B-batch_size(sequence);T-block_size(tokens);C-vocab_size
      logits=logits.view(B*T,C)                #flattening logits to a 2d array
      targets=targets.view(B*T)                #flatteing targets to a 1d array
      loss=F.cross_entropy(logits,targets)     #internally does softmax and negative log computing
    return logits,loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond=idx[:,-block_size:]
      logits,loss=self(idx_cond)
      logits=logits[:,-1,:]                    #this takes you to the last token so far, so we can continue prediction from there
                                             #logits is still 3d tensor as no targets have been provided
      probs=F.softmax(logits, dim=-1)                  #finds probablilities
      idx_next=torch.multinomial(probs,num_samples=1)  #weighted random sampling
      idx=torch.cat((idx,idx_next),dim=1)              #concatenation
    return idx

model=BigramLanguageModel()
m=model.to(device)
#expected loss: -ln(1/65)=4.1744

optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter%eval_interval==0:
    losses=estimate_loss()
    print(f"step(iter): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb,yb=get_batch('train')

  logits,loss=m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context=torch.zeros((1,1), dtype=torch.long,device=device)
generated_text = decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long,device=device),max_new_tokens=10000)[0].tolist())
with open('generated_output.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
print(generated_text)

