"""Production: exactly two visible GPUs, TF2.15.1, global batch16."""
import tensorflow as tf
_strategy=None

def get_strategy():
    global _strategy
    if _strategy is None:
        if tf.__version__ != '2.15.1':
            raise RuntimeError('V5 requires TensorFlow2.15.1 on both models')
        gpus=tf.config.list_physical_devices('GPU')
        if len(gpus)!=2:
            raise RuntimeError('V5 requires exactly 2 scheduler-visible GPUs; request --gres=gpu:2')
        for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
        _strategy=tf.distribute.MirroredStrategy()
        if _strategy.num_replicas_in_sync!=2:raise RuntimeError('Expected two replicas')
        print('V5: MirroredStrategy; global batch16; per-GPU full batch8; synchronized BatchNorm',flush=True)
    return _strategy


def balanced_loss(per_example_loss):
    """Preserve a global per-record mean even for unequal final replica batches.

    Keras averages each replica locally and then divides by replica count.
    Correct its local contribution by R*n_local/n_global; no samples dropped.
    """
    def loss(y_true,y_pred):
        values=per_example_loss(y_true,y_pred)
        context=tf.distribute.get_replica_context()
        if context is not None:
            local_n=tf.cast(tf.shape(values)[0],values.dtype)
            global_n=context.all_reduce(tf.distribute.ReduceOp.SUM,local_n)
            values=values*tf.math.divide_no_nan(local_n*tf.cast(context.num_replicas_in_sync,values.dtype),global_n)
        return values
    return loss
