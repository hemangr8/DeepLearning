import pickle

import pandas as pd
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.models import Model
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import LSTM, Lambda
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os


# Dictionary for handling one-hot-encoding of reviews.
# Can be done through tf,one_hot

dictionary = {
    1: [1, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0],
    3: [0, 0, 1, 0, 0],
    4: [0, 0, 0, 1, 0],
    5: [0, 0, 0, 0, 1]
}


# Make the input of embedded characters binary
def make_binary(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

  
# Make the outbut of lambda of embedded characters binary
def make_out_shape_binary(in_shape):
    return in_shape[0], in_shape[1], 71


# Clean the data
def remove_non_ascii(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


# Loss history for Tensorboard in callback
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


# Root path of file for training
path = "/path/to/files"

train_df = pd.read_csv(path + "/train.csv")

train_df.columns = ['id', 'AppVersionCode', 'AppVersionName', 'ReviewText',
                    'ReviewTitle', 'StarRating']

# Data cleaning and adding the review and title fiend to one text field
train_df["TextField"] = train_df.apply(lambda x: re.sub('[^a-z0-9\s]', '', ' '.join([str(x.ReviewTitle), str(x.ReviewText)]).lower().strip().replace('nan', '')) if not isinstance(x.ReviewTitle,float) or not isinstance(x.ReviewText, float) else '', axis=1)
train_df["TextField"] = train_df.apply(lambda x: re.sub('\s+', ' ', x.TextField), axis=1)

# Get a list of items in corpus
text = ''
sentences = []
words = []
stars = []

for review, star in zip(train_df.TextField, train_df.StarRating):
    words = re.split(r'\s+', remove_non_ascii(review))
    words = [sent.lower() for sent in words]
    sentences.append(words)
    stars.append(star)

# For making a set of unique characters
for sentence in sentences:
    for s in sentence:
        text += s

chars = set(text)

# Embeddings/Featurization
char_to_indices = dict((c, i) for i, c in enumerate(chars))

# Dump for inference
pickle.dump(char_to_indices, open(path + "/mapping.pkl", 'wb'), protocol=4)

max_len = 256
max_words = 256

# Initialisation of X set
X = np.ones((len(sentences), max_words, max_len), dtype=np.int64) * -1
y_raw = np.array(stars)

# Initialisation of y set
y = []

for item in y_raw:
    y.append(dictionary[item])

y = np.array(y)

# Creating X set from character embeddings of every sentence
for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        if j < max_words:
            for t, char in enumerate(word[-max_len:]):
                X[i, j, (max_len - 1 - t)] = char_to_indices[char]

# Shuffle
ids = np.arange(len(X))
np.random.shuffle(ids)

X = X[ids]
y = y[ids]

# Slpitting in train and test
X_train = X[:5000]
X_test = X[5000:]

y_train = y[:5000]
y_test = y[5000:]

# CNN filters size and dimensions
cnn_filters_length = [5, 3, 3]
nb_filter = [196, 196, 256]
cnn_pool_length = 2


input_layer_sentence = Input(shape=(max_words, max_len), dtype='int64')

# Encoder side with word encoded in character embeddings along with CNN
input_layer_word = Input(shape=(max_len,), dtype='int64')

chars_embedded = Lambda(make_binary, output_shape=make_out_shape_binary)(input_layer_word)

for i in range(len(nb_filter)):
    chars_embedded = Conv1D(filters=nb_filter[i],
                            kernel_size=cnn_filters_length[i],
                            padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            strides=1)(chars_embedded)

    chars_embedded = Dropout(0.1)(chars_embedded)
    chars_embedded = MaxPooling1D(pool_size=cnn_pool_length)(chars_embedded)

# Shallow attention to word
bi_lstm_word = Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(chars_embedded)

word_encode = Dropout(0.3)(bi_lstm_word)

encoder = Model(inputs=input_layer_word, outputs=word_encode)
encoder.summary()

# Classifier with sentence shallow attention
encoded = TimeDistributed(encoder)(input_layer_sentence)

bi_lstm_sentence = Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(encoded)

out = Dropout(0.3)(bi_lstm_sentence)
out = Dense(128, activation='relu')(out)
out = Dropout(0.3)(out)
out = Dense(5, activation='softmax')(out)

model = Model(inputs=input_layer_sentence, outputs=out)

model.summary()

file_name = 'model'
check_callback = keras.callbacks.ModelCheckpoint(path + file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor='val_loss',
                                                 verbose=0, save_best_only=True, mode='min')
early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = LossHistory()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.25, batch_size=10,
          epochs=5, shuffle=True, callbacks=[early_stop_callback, check_callback, reduce_lr_loss, history])

model.save(path + '/model_new.h5')

# Evaluate
model.evaluate(X_test, y_test, verbose=1)

