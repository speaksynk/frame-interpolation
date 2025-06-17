




import numpy as np
import tensorflow as tf
# import tensorflow_neuronx as tfnx

from eval.util import read_image


model_path="./film_model"

_model = tf.compat.v2.saved_model.load(model_path)


frame_1_path = "./photos/one.png"
frame_2_path = "./photos/two.png"

frame1 = read_image(frame_1_path)
frame2 = read_image(frame_2_path)


time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

inputs = {'x0': frame1[np.newaxis, ...], 'x1': frame2[np.newaxis, ...], 'time': time[..., np.newaxis]}


# test on CUDA ////////////////////////////////////////////////////
result = _model(inputs, training=False)

#//////////////////////////////////////////////////////////////////


# model_neuron = tfnx.trace(_model, inputs)

# print(model_neuron.on_neuron_ratio)

# model_dir = './model_neuron'
# model_neuron.save(model_dir)
# model_neuron_reloaded = tf.keras.models.load_model(model_dir)
