#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F

with open("Transformers/input.txt") as f:
  text = f.read()

#-------------HYPERPARAMS------------------#
TRAIN_SIZE = 0.8
CONTEXT_LENGTH = 8
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_DIM = 32
VOCAB_SIZE = len(set(text))
TRANSFORMER_BLOCKS = 2
ATTENTION_HEADS = 4
MAX_ITERS = 2000
LEARNING_RATE = 3e-4


#-------------TRANSFORMER MODEL------------------#
class FFNN(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.l1 = nn.Linear(emb_dim, emb_dim * 4)
    self.l2 = nn.Linear(emb_dim * 4, emb_dim)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    out1 = F.relu(self.l1(x))
    out2 = F.relu(self.l2(out1))
    out3 = self.dropout(out2)

    return out3


class AttentionHead(nn.Module):
  def __init__(self, head_len):
    super().__init__()
    self.head_len = head_len
    self.q = nn.Linear(EMBEDDING_DIM, head_len)
    self.k = nn.Linear(EMBEDDING_DIM, head_len)
    self.v = nn.Linear(EMBEDDING_DIM, head_len)

  def forward(self, x):
    queries = self.q(x)
    keys    = self.k(x)
    values  = self.v(x)

    qkt   = queries @ torch.transpose(keys, dim0=1, dim1=2)
    qkt  /= torch.sqrt(torch.tensor(self.head_len))

    qkt = torch.tril(qkt)
    qkt[qkt == 0] = float("-inf")

    probs = F.softmax(qkt, dim=-1)

    out2 = probs @ values
    return out2


class SelfAttention(nn.Module):
  def __init__(self):
    super().__init__()
    head_len = EMBEDDING_DIM // ATTENTION_HEADS
    self.att = nn.ModuleList([
        AttentionHead(head_len) for i in range(ATTENTION_HEADS)
    ])
    self.proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

  def forward(self, x):
    out1 = torch.cat([att(x) for att in self.att], dim=-1)
    return self.proj(out1)


class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.att = SelfAttention()
    self.layer_norm_1 = nn.LayerNorm(EMBEDDING_DIM)
    self.layer_norm_2 = nn.LayerNorm(EMBEDDING_DIM)
    self.ffnn = FFNN(EMBEDDING_DIM)

  def forward(self, x):
    x = x + self.att(self.layer_norm_1(x))
    x = x + self.ffnn(self.layer_norm_2(x))

    return x


class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb     = nn.Embedding(VOCAB_SIZE,     EMBEDDING_DIM)
    self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)

    self.transformer_blocks = nn.Sequential(
        *[Block() for _ in range(TRANSFORMER_BLOCKS)]
    )

    self.lin = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

  def forward(self, x):
    emb = self.emb(x) + self.pos_emb(torch.arange(0, x.shape[1]).to(DEVICE))
    transformer_out = self.transformer_blocks(emb)
    logits = self.lin(transformer_out)

    return logits


#-------------DATA PREPROCESSING------------------#
chars = set(text)

stoi = {val:idx for (idx, val) in enumerate(chars)}
itos = {val:key for (key, val) in stoi.items()}

encode = lambda x: [stoi[i] for i in x]
decode = lambda x: [itos[i] for i in x]

tokenized_text = torch.tensor(encode(text))

train_data = tokenized_text[:int(len(tokenized_text) * TRAIN_SIZE)]
test_data  = tokenized_text[int(int(len(tokenized_text) * TRAIN_SIZE)):]

train_data.to(DEVICE)
test_data.to(DEVICE)


def get_batch(data):
  indices = torch.randint(high=data.shape[0] - CONTEXT_LENGTH - 1, size=(BATCH_SIZE,))
  X_batch = torch.stack([data[i:i + CONTEXT_LENGTH].clone().detach()       for i in indices])
  Y_batch = torch.stack([data[i+1:i + CONTEXT_LENGTH + 1].clone().detach() for i in indices])

  return (X_batch.to(DEVICE), Y_batch.to(DEVICE))

#-------------MODEL TRAINING------------------#
def calc_loss(model, optim):
  X_train, y_train = get_batch(train_data)
  X_test, y_test   = get_batch(test_data)

  X_train = X_train.to(DEVICE)
  y_train = y_train.to(DEVICE)
  X_test  = X_test.to(DEVICE)
  y_test  = y_test.to(DEVICE)

  output_train  = model(X_train)
  output_test   = model(X_test)
  B, T, C = output_train.shape

  train_loss = F.cross_entropy(output_train.reshape((B * T, C)),
                                         y_train.reshape(B * T))

  with torch.no_grad():
    test_loss  = F.cross_entropy(output_test.reshape((B * T, C)),
                                          y_test.reshape(B * T))

  return (train_loss, test_loss)


transformer = Transformer()
transformer = transformer.to(DEVICE)
optim = torch.optim.AdamW(transformer.parameters(), 1e-3)


train_lossi = []
test_lossi = []
for iter in range(MAX_ITERS):
  if iter % 200 == 0 and iter > 0:
    print(f"ITERATIONS: {iter}  |  TRAIN LOSS: {train_loss:.4f}  |  TEST LOSS: {test_loss:.4f}")

  train_loss, test_loss = calc_loss(transformer, optim)

  optim.zero_grad()
  train_loss.backward()
  optim.step()

  train_lossi.append(train_loss)
  test_lossi.append(test_loss)

print(f"-------------------TRANSFORMER TRAINING COMPLETE-------------------\n\n")


#-------------GENERATE TEXT------------------#
x = torch.ones((1, 1), dtype=torch.long).to(DEVICE)
max_new = 1000

for _ in range(max_new):
  output = transformer(x[:, -CONTEXT_LENGTH:])
  logits = output[:, -1, :]
  probs = F.softmax(logits, dim=-1)
  x_next = torch.multinomial(probs, num_samples=1)
  x = torch.cat((x, x_next), dim=1)

print(''.join(decode(x[0].detach().numpy())))