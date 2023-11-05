import os

import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('my_model.keras')
#input_text = "Hello world"
#output = model(input_text)
print(model)

