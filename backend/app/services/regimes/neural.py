"""CPU LSTM/GRU training with learned recurrent weights and chronological sequences."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from app.core.errors import ValidationFailure

# Torch's RNG is process-wide; serialize seeded training, not inference.
_TRAIN_LOCK = threading.Lock()


def network(kind, inputs, hidden, layers, classes):
    import torch

    class Recurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cell = torch.nn.LSTM if kind == "lstm" else torch.nn.GRU
            self.recurrent = cell(inputs, hidden, layers, batch_first=True)
            self.readout = torch.nn.Linear(hidden, classes)

        def forward(self, x):
            sequence, _ = self.recurrent(x)
            return self.readout(sequence[:, -1])

    return Recurrent()


def sequences(x, indices, length):
    if len(indices) == 0 or indices.min() < length - 1:
        raise ValidationFailure("Insufficient causal history for the configured sequence length.")
    return np.stack([x[i - length + 1 : i + 1] for i in indices]).astype(np.float32)


@dataclass
class Neural:
    kind: str
    parameters: dict
    inputs: int
    classes_: np.ndarray
    weights: dict
    history: list[float]

    def predict_proba(self, x):
        import torch

        model = network(
            self.kind,
            self.inputs,
            int(self.parameters["hidden_size"]),
            int(self.parameters["layers"]),
            len(self.classes_),
        )
        model.load_state_dict(
            {key: torch.from_numpy(value.copy()) for key, value in self.weights.items()}
        )
        model.eval()
        length = int(self.parameters["sequence_length"])
        data = sequences(x, np.arange(length - 1, len(x)), length)
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.from_numpy(data)), dim=1).numpy()
        # Unavailable warm-up rows are explicit, not fabricated zero-probabilities.
        return np.vstack([np.full((length - 1, len(self.classes_)), np.nan), probabilities])


def fit_neural(kind, x, indices, labels, parameters, seed):
    import torch

    length = int(parameters["sequence_length"])
    mask = indices >= length - 1
    indices, labels = indices[mask], labels[mask]
    classes, encoded = np.unique(labels, return_inverse=True)
    if len(indices) < 8 or len(classes) < 2:
        raise ValidationFailure(
            "Recurrent training needs at least 8 complete sequences and two target classes."
        )
    with _TRAIN_LOCK, torch.random.fork_rng():
        torch.manual_seed(seed)
        model = network(
            kind,
            x.shape[1],
            int(parameters["hidden_size"]),
            int(parameters["layers"]),
            len(classes),
        )
        data = torch.from_numpy(sequences(x, indices, length))
        target = torch.tensor(encoded, dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
        loss_function = torch.nn.CrossEntropyLoss()
        history = []
        for _ in range(int(parameters["epochs"])):
            model.train()
            total = 0.0
            # Each sequence ends at its origin; batches do not carry recurrent state.
            for start in range(0, len(data), int(parameters["batch_size"])):
                end = start + int(parameters["batch_size"])
                optimizer.zero_grad()
                loss = loss_function(model(data[start:end]), target[start:end])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(data[start:end])
            history.append(total / len(data))
        weights = {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}
    return Neural(kind, parameters, x.shape[1], classes, weights, history)
