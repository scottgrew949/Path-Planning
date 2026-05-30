# python/train_heuristic_net.py
#
# PURPOSE: Train HeuristicNetwork on the generated h* dataset, validate it,
#          and export binary weights for C++ inference.
#
# CORE CONCEPT — Supervised learning as curve fitting
#   We have (input, label) pairs. We adjust model parameters so the model's
#   outputs match the labels as closely as possible. "Training" = running
#   gradient descent to minimise a loss function over the dataset.
#   This is mathematically identical to least-squares curve fitting — just
#   applied to a function with millions of parameters (the network weights).
#
# CORE CONCEPT — Loss function: MSE for regression
#   Classification tasks use cross-entropy loss (predicting a probability).
#   Regression tasks (predicting a continuous value like h*) use MSE:
#       L = (1/N) * sum((h_hat - h*)^2)
#   Why squared? Squaring penalises large errors more than small ones,
#   pushing the model to get the important cases right.
#   Why NOT mean absolute error (MAE)? MAE has a kink at zero — its gradient
#   is undefined at perfect prediction, making optimisation noisy near convergence.
#   MSE has a smooth, well-defined gradient everywhere.
#
# CORE CONCEPT — Mini-batch stochastic gradient descent
#   Full gradient descent computes the gradient over the ENTIRE dataset each step.
#   That's slow and uses huge memory for large datasets.
#   Stochastic GD uses ONE sample per step — fast but very noisy gradients.
#   Mini-batch GD (what we use) uses a BATCH of e.g. 256 samples.
#   The batch gradient is a noisy estimate of the true gradient, but the noise
#   actually helps — it prevents getting stuck in sharp local minima.
#   Batch size 256 is a practical sweet spot: fits in CPU cache, stable gradients.
#
# CORE CONCEPT — Train/validation split
#   We hold out 20% of data as a validation set — the model NEVER trains on it.
#   Each epoch we measure loss on BOTH train and val sets.
#   If train loss drops but val loss rises: the model is OVERFITTING.
#   It's memorising the training mazes, not learning a general heuristic.
#   Early stopping (stop when val loss stops improving) prevents this.
#
# CORE CONCEPT — Adam optimiser
#   Adam (Adaptive Moment Estimation) is the default choice for neural net training.
#   It adapts the learning rate per-parameter using estimates of gradient mean
#   and variance. Concretely: parameters with consistently large gradients get
#   smaller step sizes; rarely updated parameters get larger steps.
#   This makes it robust to different feature scales and sparse gradients.
#   lr=1e-3 is the Adam default and almost always a good starting point.
#
# CONCEPT — Admissibility check (weighted A*)
#   After training, we check whether h_hat ever overestimates h*.
#   For weighted A* with ε=1.5: we allow h_hat ≤ 1.5 * h*.
#   We measure the EMPIRICAL suboptimality ratio = max(h_hat / h*) on the val set.
#   This tells us what ε guarantee we can claim — not just hope for.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os

from heuristic_net import HeuristicNetwork, export_weights

DATA_PATH    = os.path.join(os.path.dirname(__file__), 'data', 'heuristic_training.npy')
WEIGHTS_PATH    = os.path.join(os.path.dirname(__file__), 'data', 'weights.bin')
WEIGHTS_PT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'weights.pt')

BATCH_SIZE   = 256
NUM_EPOCHS   = 60
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2


def load_dataset(data_path: str) -> tuple:
    """
    CONCEPT — Tensor creation from numpy:
    PyTorch operates on Tensors, not numpy arrays. torch.from_numpy() shares
    memory — no copy. .float() converts float64 → float32 because PyTorch
    nn.Linear uses float32 by default (faster, uses less memory).

    Implement:
    1. Load npy file with np.load()
    2. Split columns: inputs = data[:, :4], labels = data[:, 4:]
    3. Convert to float32 tensors with torch.tensor(..., dtype=torch.float32)
    4. Return (inputs_tensor, labels_tensor)
    """
    data   = np.load(data_path)
    inputs = torch.tensor(data[:, :4], dtype=torch.float32)
    labels = torch.tensor(data[:, 4:], dtype=torch.float32)
    return (inputs, labels)


def build_dataloaders(inputs: torch.Tensor, labels: torch.Tensor) -> tuple:
    """
    CONCEPT — DataLoader as curriculum:
    DataLoader handles batching and shuffling. shuffle=True on train set means
    each epoch sees data in a different order — prevents the model from learning
    the ordering of samples rather than the underlying function.
    The val DataLoader does NOT shuffle (order doesn't matter for evaluation).

    Implement:
    1. Wrap inputs + labels in TensorDataset
    2. Split into train/val with random_split() using VAL_FRACTION
    3. Return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            and DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    """
    dataset    = TensorDataset(inputs, labels)
    val_size   = int(len(dataset) * VAL_FRACTION)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)


def train_one_epoch(
    model: HeuristicNetwork,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function: nn.Module
) -> float:
    """
    CONCEPT — Training loop mechanics:
    Each batch goes through four steps:
    1. FORWARD:  output = model(inputs)           — compute predictions
    2. LOSS:     loss   = loss_fn(output, labels) — measure error
    3. BACKWARD: loss.backward()                  — compute gradients via autograd
    4. STEP:     optimiser.step()                 — update weights

    optimiser.zero_grad() MUST come before backward(). Why?
    PyTorch ACCUMULATES gradients by default (adds to existing .grad tensor).
    If you forget zero_grad(), gradients from previous batches contaminate
    the current batch update. This is a very common bug.

    model.train() enables dropout and batch norm (not used here, but good habit).

    Implement:
    1. model.train()
    2. For each (batch_inputs, batch_labels) in train_loader:
       a. optimiser.zero_grad()
       b. predictions = model(batch_inputs)
       c. loss = loss_function(predictions, batch_labels)
       d. loss.backward()
       e. optimiser.step()
       f. accumulate loss.item() * batch size
    3. Return mean loss over all samples (accumulated_loss / total_samples)
    """
    model.train()
    accumulated_loss  = 0.0
    total_samples     = 0

    for batch_inputs, batch_labels in train_loader:
        optimiser.zero_grad()
        predictions = model(batch_inputs)
        loss        = loss_function(predictions, batch_labels)
        loss.backward()
        optimiser.step()
        accumulated_loss += loss.item() * batch_inputs.shape[0]
        total_samples    += batch_inputs.shape[0]

    return accumulated_loss / total_samples


def evaluate(
    model: HeuristicNetwork,
    val_loader: DataLoader,
    loss_function: nn.Module
) -> tuple:
    """
    CONCEPT — Evaluation without gradient tracking:
    torch.no_grad() disables autograd — no computation graph is built.
    This halves memory usage and speeds up inference by ~30%.
    During evaluation we are NOT updating weights, so we don't need gradients.
    model.eval() disables dropout layers (ensures deterministic output).

    Also compute the admissibility ratio: max(h_hat / h*) across val set.
    This tells us the worst-case overestimation factor.
    For weighted A* with ε=1.5 to be valid, this ratio must be ≤ 1.5.

    Implement:
    1. model.eval()
    2. with torch.no_grad():
       For each (batch_inputs, batch_labels) in val_loader:
         predictions = model(batch_inputs)
         accumulate MSE loss
         compute ratio = predictions / (batch_labels + 1e-8)  ← epsilon avoids div/0
         track max ratio seen so far
    3. Return (mean_val_loss, max_admissibility_ratio)
    """
    model.eval()
    accumulated_loss   = 0.0
    total_samples      = 0
    max_ratio          = 0.0

    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            predictions       = model(batch_inputs)
            loss              = loss_function(predictions, batch_labels)
            accumulated_loss += loss.item() * batch_inputs.shape[0]
            total_samples    += batch_inputs.shape[0]
            ratio             = (predictions / (batch_labels + 1e-8)).max().item()
            if ratio > max_ratio:
                max_ratio = ratio

    return (accumulated_loss / total_samples, max_ratio)


def main():
    """
    CONCEPT — Full training pipeline:
    The outer loop (epochs) sweeps through the dataset repeatedly.
    The inner loop (batches) processes one mini-batch per iteration.
    After each epoch: print train loss, val loss, and admissibility ratio.
    These three numbers tell you:
      - train loss: is the model learning at all?
      - val loss:   is it generalising or memorising?
      - ratio:      is the heuristic admissible enough for our ε?

    CONCEPT — When to stop training:
    Stop when val loss plateaus (stops decreasing for ~10 epochs).
    Don't stop based on train loss alone — train loss always decreases.
    The gap between train and val loss diagnoses overfitting.

    Implement:
    1. load_dataset() → inputs, labels
    2. build_dataloaders() → train_loader, val_loader
    3. Instantiate HeuristicNetwork(), nn.MSELoss(), torch.optim.Adam(lr=LEARNING_RATE)
    4. Loop NUM_EPOCHS:
       a. train_one_epoch() → train_loss
       b. evaluate() → val_loss, admissibility_ratio
       c. Print epoch summary: "Epoch {n}: train={:.4f} val={:.4f} max_ratio={:.3f}"
    5. After training:
       a. export_weights(model, WEIGHTS_PATH)          — binary format for C++ inference
       b. torch.save(model.state_dict(), WEIGHTS_PT_PATH) — PyTorch format for visualisation
       Both must be saved: C++ reads .bin, heuristic_quality.py reads .pt
    6. Print final admissibility summary:
       "Max h_hat/h* = {ratio:.3f} — weighted A* with ε≥{ratio:.2f} is valid"
    """
    inputs, labels   = load_dataset(DATA_PATH)
    train_loader, val_loader = build_dataloaders(inputs, labels)
    print(f"Dataset: {len(inputs):,} samples  "
          f"train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}")

    model         = HeuristicNetwork()
    loss_function = nn.MSELoss()
    optimiser     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    max_ratio = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss              = train_one_epoch(model, train_loader, optimiser, loss_function)
        val_loss, max_ratio     = evaluate(model, val_loader, loss_function)
        print(f"Epoch {epoch:3d}: train={train_loss:.6f}  val={val_loss:.6f}  "
              f"max_ratio={max_ratio:.3f}")

    export_weights(model, WEIGHTS_PATH)
    torch.save(model.state_dict(), WEIGHTS_PT_PATH)
    print(f"Weights saved → {WEIGHTS_PATH}")
    print(f"Max h_hat/h* = {max_ratio:.3f} — "
          f"weighted A* with ε≥{max_ratio:.2f} is valid")


if __name__ == '__main__':
    main()
