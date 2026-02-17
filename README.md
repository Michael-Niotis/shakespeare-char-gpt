# Shakespeare Text Generator — Character-Level GPT (PyTorch)

A GPT-style **decoder-only Transformer** trained at the **character level** on the Shakespeare corpus to generate Shakespeare-like text from a seed prompt.

## Highlights

- Implemented a **decoder-only Transformer from scratch** in PyTorch (causal multi-head self-attention, pre-LN blocks, residual connections, dropout).
- Built a **character-level data pipeline** (vocabulary, tokenization, sliding-window batching with fixed context length).
- Added training stabilizers: **gradient clipping**, **learning-rate decay**, and **dropout tuning**.
- Implemented **checkpointing** (`ckpt.pt`) and a separate **inference script** (`generate.py`) to generate text without retraining.
- Logged **train vs validation loss** to diagnose overfitting on a small corpus and guide regularization choices.

## Repository contents

- `shakespear.py` — end-to-end training + validation + checkpointing + loss plot + sample generation
- `generate.py` — loads `ckpt.pt` and generates text
- `input.txt` — Shakespeare dataset (plain text)
- `Losses.png` — training vs validation loss plot

## Quickstart
Run training first to create `ckpt.pt`, then run `generate.py`

### Install
```bash
pip install torch matplotlib
```

### Train
```bash
python shakespear.py
```
This will:
- build the character vocabulary from `input.txt`
- train for `train_steps`
- save a checkpoint to `ckpt.pt`
- plot training/validation losses
- generate a sample continuation from the default prompt

### Generation (without retraining)
After ckpt.pt exists:
```bash
python generate.py
```

## What this project does

- Trains an autoregressive Transformer to predict the **next character**
- Uses **causal self-attention** (masking future tokens) so the model cannot "peek" ahead
- Generates text from a prompt (default: `"O God, O God!"`)

## Model architecture (decoder-only Transformer)

Implemented in PyTorch with:
- token embeddings + positional embeddings
- stacked Transformer blocks (pre-layernorm)
  - causal multi-head self-attention
  - MLP / feed-forward network
  - residual connections + dropout
- final LayerNorm + output projection to vocabulary logits

## Default hyperparameters

- `block_size = 128` (context length)
- `batch_size = 128`
- `nb_layers = 12`
- `nb_heads = 8`
- `nb_embd = 768`
- `lr = 5e-4`
- LR schedule: `StepLR(step_size=1000, gamma=0.7)`
- `train_steps = 6000`
- `residual_pdrop = 0.1`
- `embd_pdrop = 0.2`
- Gradient clipping: clip when gradient L2 norm > 1.0


## Results (from experiments)

### Training curves
- training loss converged to ~**0.155**
- validation loss around **3.26** (after 6000 steps with LR decay + gradient clipping + increased embedding dropout)
![Training vs Validation Loss](Losses.png)

### Notes on generalization
On this small dataset, validation loss eventually increases while training loss continues to decrease (overfitting).

The improved setup focused on training stability and generation quality and diversity, not only minimizing validation loss.

## Experiment iteration (exploratory baseline → current configuration)

I first ran an **exploratory baseline** and logged train/validation losses, then updated the configuration to improve training stability and generation quality.  
Only the **current configuration** is encoded in the repository scripts (`shakespear.py`, `generate.py`). The baseline run was exploratory.

| Setup | Decoding | Key settings | Train loss (≈) | Val loss (≈) |
|------|----------|--------------|----------------|--------------|
| Baseline (exploratory) | Greedy (argmax) | Adam + Cross-Entropy, lr=3e-4 (constant), embd dropout=0.1, no LR decay, no grad clipping | 0.19 | 3.0 |
| Current configuration (repo) | Multinomial sampling | Adam + Cross-Entropy, lr=5e-4 + StepLR (γ=0.7 every 1000 steps), embd dropout=0.2, grad clip (L2 norm > 1) | 0.155 | 3.26 |

## Decoding comparison: greedy (argmax) vs sampling (multinomial)

I compared two decoding strategies:

- **Greedy / argmax (deterministic):** always chooses the most likely next character → can be repetitive or jump abruptly.
- **Multinomial sampling (stochastic):** samples from the model distribution → more diverse generations.

### Greedy (argmax) sample
```text
O God, O God! that e'er the sun's beat,
And he shall marry her: the nuptial finish'd,
Let him be whipt and hang'd.

LUCIO: I beseech you, sir, look in gentle words,
For many more evasion: yet this love we meet,
That slew thy words are and stately both.

SICINIUS: March on, my fellows:
Make good this ostentation, and you shall
Divide in all with us.
```

### Multinomial sample
```text
O God, O God! that e'er this tongue of mine,
That laid the sentence of dread banishment
On yon proud man, should take it off again
With words of sooth! O that I were as great
As is my grief, or lesser than my name!
Or that I could forget what I have been,
Or not remember what I must be now!
Swell'st thou, proud heart? I'll give thee scope to beat,
Since foes have scope to beat both thee and me.

DUKE OF AUMERLE: Northumberland comes back from Bolingbroke.
KING RICHARD II: What must the king do now? must he submit?
The king shall do it: must he be deposed?
```




## Requirements

- Python 3.9+ recommended
- PyTorch
- matplotlib


