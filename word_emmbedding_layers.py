# Word Embedding Layers for Deep Learning with Keras

# Import libraries
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

# sentences
sent = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# Vocabulary size
voc_size = 1000

# One Hot Representation
one_hot_repr = [one_hot(word, voc_size) for word in sent]

print(one_hot_repr)

# Word Embedding Representation
sent_length = 8
embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

dim = 10

# Define model
model = Sequential()
model.add(Embedding(voc_size, 10, input_length=sent_length))
print(model.summary())


print(model.predict(embedded_docs))


embedded_docs[0]


print(model.predict(embedded_docs)[0])