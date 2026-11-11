"""Replica-safe batching shared by every deep-learning model.

Problem fixed: with global batch 16 on two replicas, a final batch of 1 record
leaves one replica with an EMPTY batch (fold 1 of V5: 19905 % 16 == 1) and the
run diverged to NaN.  Rule applied identically to FAT, Uniform and Preaggregation:
batches are 16 records; if the remainder is smaller than the number of replicas
(only remainder 1 with 2 GPUs) the last 16+r records are split into two nearly
equal batches (9+8).  No record is dropped, every step has <=16 records and the
number of steps per epoch is unchanged (ceil(n/16)).
"""
import numpy as np


def batch_sizes(n, batch=16, replicas=2):
    n = int(n)
    full, rem = divmod(n, batch)
    sizes = [batch] * full
    if rem:
        if rem < replicas and full > 0:
            total = batch + rem
            first = (total + 1) // 2
            sizes[-1] = first
            sizes.append(total - first)
        else:
            sizes.append(rem)
    assert sum(sizes) == n
    return sizes


def _sequence_base():
    import tensorflow as tf
    return tf.keras.utils.Sequence


def make_batches(xs, y, sizes, shuffle, seed=0):
    base = _sequence_base()

    class EpochBatches(base):
        """Host-NumPy batches; optional per-epoch full permutation (like Keras shuffle=True)."""

        def __init__(self):
            super().__init__()
            self.xs = xs
            self.y = y
            self.sizes = list(sizes)
            self.starts = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
            self.n = int(self.starts[-1])
            self.shuffle = shuffle
            self.rng = np.random.RandomState(seed)
            self.order = self.rng.permutation(self.n) if shuffle else np.arange(self.n)

        def __len__(self):
            return len(self.sizes)

        def __getitem__(self, i):
            idx = self.order[self.starts[i]:self.starts[i + 1]]
            if not self.shuffle:
                idx = slice(int(idx[0]), int(idx[-1]) + 1)
            batch = [np.ascontiguousarray(a[idx]) for a in self.xs]
            if self.y is None:
                return (batch,)
            return batch, np.ascontiguousarray(self.y[idx])

        def on_epoch_end(self):
            if self.shuffle:
                self.order = self.rng.permutation(self.n)

    return EpochBatches()


def n_replicas(model):
    """Replica count of the strategy the model was built under (works outside the strategy scope)."""
    return int(model.distribute_strategy.num_replicas_in_sync)


def predict_in_batches(model, xs, batch=16, verbose=2):
    """Inference with the same replica-safe batch plan (BN uses moving statistics, so results are batch-independent).

    Uses predict_on_batch (same mechanism as train_on_batch): Keras 2.15's Sequence adapter mis-handles
    x-only batches with several inputs.
    """
    sizes = batch_sizes(len(xs[0]), batch, n_replicas(model))
    out = []
    start = 0
    for i, size in enumerate(sizes):
        out.append(np.asarray(model.predict_on_batch([np.ascontiguousarray(a[start:start + size]) for a in xs])))
        start += size
        if verbose and (i + 1) % 100 == 0:
            print('predict %d/%d' % (i + 1, len(sizes)), flush=True)
    return np.concatenate(out, axis=0)


def fit_with_safe_batches(model, xs, y, epochs, batch_size, validation_split, callbacks):
    """Keras-equivalent fit: last `validation_split` fraction held out in order, per-epoch shuffle of the rest."""
    import tensorflow as tf
    n = len(y)
    split_at = int(n * (1 - validation_split)) if validation_split > 0 else n
    reps = n_replicas(model)
    seed = int(np.random.randint(0, 2 ** 31 - 1))  # drawn from the fold-seeded NumPy stream
    train_xs = [a[:split_at] for a in xs]
    train = make_batches(train_xs, y[:split_at], batch_sizes(split_at, batch_size, reps), True, seed)
    val = None
    if validation_split > 0 and split_at < n:
        val_xs = [a[split_at:] for a in xs]
        val = make_batches(val_xs, y[split_at:], batch_sizes(n - split_at, batch_size, reps), False)
    guard = FiniteLossGuard()
    return model.fit(train, validation_data=val, epochs=epochs, verbose=2, shuffle=False,
                     callbacks=list(callbacks) + [guard])


def _guard_class():
    import tensorflow as tf

    class FiniteLossGuard(tf.keras.callbacks.Callback):
        """Fail immediately (with the step index) instead of training 30 NaN epochs."""

        def on_train_batch_end(self, batch, logs=None):
            loss = (logs or {}).get('loss')
            if loss is not None and not np.isfinite(loss):
                raise RuntimeError('Nonfinite training loss at epoch-step %d (loss=%r)' % (batch, loss))

    return FiniteLossGuard


def FiniteLossGuard():
    return _guard_class()()
