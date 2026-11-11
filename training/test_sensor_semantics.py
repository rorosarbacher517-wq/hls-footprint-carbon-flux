import json
from pathlib import Path
import numpy as np,tensorflow as tf
ROOT=Path(__file__).resolve().parent
report={'tensorflow':tf.__version__,'raw_input_categories':[0,1]}
for device in ['/CPU:0','/GPU:0']:
    if device=='/GPU:0' and not tf.config.list_physical_devices('GPU'):continue
    with tf.device(device):
        layer=tf.keras.layers.Embedding(2,2,embeddings_initializer=tf.keras.initializers.Constant([[1.,2.],[3.,4.]]))
        with tf.GradientTape() as tape:
            output=layer(np.array([[0.,1.]],np.float32));loss=tf.reduce_sum(output)
        grad=tape.gradient(loss,layer.trainable_variables)[0]
        np.testing.assert_array_equal(output.numpy(),[[[1.,2.],[3.,4.]]])
        assert np.all((grad.indices.numpy()>=0)&(grad.indices.numpy()<2))
        report[device]={'lookup_correct':True,'gradient_indices':grad.indices.numpy().tolist()}
(ROOT/'audit/sensor_embedding_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
