import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

residual_pdrop = 0.1  # Attention is All You Need Paper
embd_pdrop = 0.1
block_size = 128
batch_size = 128
nb_layers = 12
nb_heads = 8
nb_embd = 768
lr = 3e-4
train_steps = 6000


class CharDataset(Dataset):
    """Emits batches of characters

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, data, block_size):

        self.block_size = block_size

        chars = sorted(list(set(data)))  # get characters from the input data

        self.stoi = {
            ch: i for i, ch in enumerate(chars)
        }  # mapping character to integer indeces
        self.itos = {
            i: ch for i, ch in enumerate(chars)
        }  # mapping integer indices to characters

        vocab_size = len(chars)
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return (
            len(self.data) - self.block_size
        )  # number of possible starting positions for a chunk of size block_size + 1

    def __getitem__(self, idx):
        chunk = self.data[
            idx : idx + self.block_size + 1
        ]  # grab a chunk of block_size + 1 characters from the data

        chunk_dict = [
            self.stoi[ch] for ch in chunk
        ]  # encode every character to an integer

        input_t = torch.tensor(chunk_dict[:-1])
        target_t = torch.tensor(chunk_dict[1:])

        return input_t, target_t  # return the chunk and the shifted version as tensors


class CausalSelfAttention(nn.Module):
    def __init__(self, nb_embd, nb_heads, block_size, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.nb_heads = nb_heads

        self.residual_pdrop = residual_pdrop
        self.block_size = block_size

        # nb_embd % nb_heads == 0
        self.attention = nn.Linear(
            self.nb_embd, 3 * self.nb_embd
        )  # I get for each token in input a vector of size 3*nb_embd
        # A big Weight matrix of shape [3*nb_embd,nb_embd] that can be considered
        # the concatenation (by last dimension) of Wq,Wk,Wv

        self.out_proj = nn.Linear(
            self.nb_embd, self.nb_embd
        )  # the output projection W0 according to paper

        # Apply dropout to the output according to "Attention is all you need" before normalization and residual connection
        self.residual_dropout = nn.Dropout(self.residual_pdrop)

        # I create the mask for causal self attention
        # with a lower triangular matrix so that no information from future flows to the past
        # in each position i, only elements in positions <= i are not zero
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )

    def forward(self, x):
        B, N, C = x.size()
        h = self.nb_heads

        qkv = self.attention(x)
        q, k, v = qkv.chunk(
            3, dim=-1
        )  # I split the big matrix, into 3 separate Q,K,V matrices of shape [B,N,C]
        # The q,k,v matrices are calculated for all heads in batch at once , shape [B,N,C]
        q = q.view(B, N, h, C // h).transpose(
            1, 2
        )  # I first split the channel dimension (nb_embd=C) into nb_heads * head_dimension
        # shape [B,heads,N(sequence),head_dimension]
        k = k.view(B, N, h, C // h).transpose(1, 2)
        v = v.view(B, N, h, C // h).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # I apply the mask for causal self-attention
        # no information from the future flows to the past
        # I use a lower triangular matrix to do this
        mask = self.mask[:, :, :N, :N]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B,heads,N,N) x (B,heads,N, head_dim) --> (B,heads,N,head_dim)
        out = (
            out.transpose(1, 2).contiguous().view(B, N, C)
        )  # merge the last 2 dimensions (heads,head_dim) into one (C = heads * head_dim) to perform the projection W0

        # perform the final projection to the concatenation of all heads
        out = self.out_proj(out)
        # perform a dropout to the output of this sub-layer according to the paper
        out = self.residual_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, nb_embd, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.residual_pdrop = residual_pdrop
        self.mlp = nn.Sequential(
            nn.Linear(self.nb_embd, 4 * self.nb_embd),
            nn.ReLU(),
            nn.Linear(4 * self.nb_embd, self.nb_embd),
        )

        self.residual_dropout = nn.Dropout(self.residual_pdrop)

    def forward(self, x):
        ffn_out = self.mlp(x)
        ffn_out = self.residual_dropout(ffn_out)
        return ffn_out


class Block(nn.Module):
    def __init__(self, nb_embd, nb_heads, block_size, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.nb_heads = nb_heads
        self.block_size = block_size
        self.residual_pdrop = residual_pdrop

        self.CausalSelfAttention = CausalSelfAttention(
            self.nb_embd, self.nb_heads, self.block_size, self.residual_pdrop
        )
        self.mlp = MLP(self.nb_embd, self.residual_pdrop)
        self.LayerNorm_1 = nn.LayerNorm(self.nb_embd)
        self.LayerNorm_2 = nn.LayerNorm(self.nb_embd)

    def forward(self, x):
        x = x + self.CausalSelfAttention(self.LayerNorm_1(x))
        x = x + self.mlp(self.LayerNorm_2(x))
        return x


class FullModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        nb_embd,
        block_size,
        nb_layers,
        nb_heads,
        embd_pdrop,
        residual_pdrop,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.nb_embd = nb_embd
        self.block_size = block_size
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.residual_pdrop = residual_pdrop
        self.embd_pdrop = embd_pdrop
        self.wte = nn.Embedding(
            self.vocab_size, self.nb_embd
        )  # I tokenize each character from our vocabulary into a vector of dimension nb_embd
        self.pte = nn.Embedding(
            self.block_size, self.nb_embd
        )  # I tokenize each position in the block_size (sequence) into a vector of dimension nb_embd
        self.dropout = nn.Dropout(
            self.embd_pdrop_pdrop
        )  # dropout to be applied in the input (sum of tok_emb and pos_emb)
        self.blocks = nn.ModuleList(
            [
                Block(self.nb_embd, self.nb_heads, self.block_size, self.residual_pdrop)
                for _ in range(self.nb_layers)
            ]
        )

        self.Final_LayerNorm = nn.LayerNorm(self.nb_embd)
        self.lm_head = nn.Linear(self.nb_embd, self.vocab_size)

    def forward(self, idx):
        B, N = idx.size()  # Batches of sequences of length N
        assert N <= self.block_size
        pos = torch.arange(0, N).unsqueeze(
            0
        )  # transform into a row vector(tensor) of shape (1,N)
        tok_emb = self.wte(idx)  # create the token embeddings, shape of B,N,nb_embd
        pos_emb = self.wpe(pos)  # positional embeddings, shape of 1,N,nb_embd
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.Final_LayerNorm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )  # Take the whole sequence if it's smaller than block_size, otherwise take the last block_size elements

            logits = self(idx_cond)  # [B,N,vocab_size]

            logits = logits[:, -1, :]  # [B,vocab_size]
            # take the last position in the sequence in order to predict the next token

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.argmax(
                probs, dim=-1, keepdim=True
            )  # [B,1] for each batch we select the element with the greatest probability

            idx = torch.cat(
                (idx_cond, next_idx), dim=1
            )  # append the index we got to the sequence
        return idx


text = open("input.txt", "r").read()
n = int(0.9 * len(text))
train_text = text[0:n]
val_text = text[n:]

train_ds = CharDataset(train_text, block_size)
val_ds = CharDataset(val_text, block_size)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True
)  # I get batches of (B,N) : B=batch_size examples(sequences) of N = block_size each
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

model = FullModel(
    train_ds.get_vocab_size,
    nb_embd,
    train_ds.get_block_size,
    nb_layers,
    nb_heads,
    embd_pdrop,
    residual_pdrop,
)
optimizer = torch.optim.Adam(model.parameters(), lr)

step = 0

while step < train_steps:
    for x, y in train_loader:
        logits = model(x)
        B, N, C = (
            logits.size()
        )  # Batch_size, position in sequence, logit scores for next position
        loss = F.cross_entropy(
            logits.view(B * N, C), y.view(B * N)
        )  # cross entropy expects an input of (Batch,Channels) and a target of of indices in the range of [0,Channels-1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 200 == 0:
            print(step, loss.item())

        if step >= train_steps:
            break
