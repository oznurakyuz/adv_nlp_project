import codecs
import os
import sys

import numpy as np
from gensim.models import fasttext
from keras import Input
from keras.layers import Embedding, Bidirectional, LSTM
from keras.preprocessing import sequence
import keras.backend as K
import numpy
# from keras.engine import Model
import tensorflow as tf
import gensim
from gensim.models import FastText
from keras import regularizers
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Reshape, Masking
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras import losses

number_of_segmentation = 10  # args.segNo
batch_size = 256  # args.batch
number_of_epoch = 5  # args.epoch
dim = 300  # args.dim

gensim_model = "../vectors/cc.tr.300.bin"  # args.wordVector
training_file = "../data_resource/training.tr"  # args.input
output_file = "../output/model.pk"  # args.output

if not os.path.exists(output_file):
    os.makedirs(output_file)

print('==========  Prepare data...  ==========')
word2sgmt = {}
word2segmentations = {}
seq = []
morphs = []
f = codecs.open(training_file, encoding='utf-8')
for line in f:
    line = line.rstrip('\n')
    word, sgmnts = line.split(':')
    sgmt = sgmnts.split('+')
    word2segmentations[word] = list(map(list, sgmt))
    sgmt = list(map(lambda s: s.split('-'), sgmt))
    word2sgmt[word] = sgmt
    seq.extend(sgmt)

timesteps_max_len = 0

for sgmt in seq:
    morphs.extend(sgmt)
    if len(sgmt) > timesteps_max_len:
        timesteps_max_len = len(sgmt)

print('number of words: ', len(word2sgmt))
print('number of morphemes: ', len(morphs))

morphs = set(morphs)
morph_indices = dict(zip(morphs, range(1, len(morphs) + 1)))
indices_morph = dict(zip(range(1, len(morphs) + 1), morphs))

morph_indices['###'] = 0
indices_morph[0] = '###'

print('number of unique morphemes: ', len(morphs))

x_train = [[] for _ in range(number_of_segmentation)]
for word in word2sgmt:
    for i in range(len(word2sgmt[word])):
        x_train[i].append([morph_indices[c] for c in word2sgmt[word][i]])

x_train = [sequence.pad_sequences(np.array(x, dtype=object), maxlen=timesteps_max_len) for x in x_train]

print('==========  Load pre-trained word vectors...  ==========')
w2v_model = fasttext.load_facebook_vectors(gensim_model)
y_train = []
for word in word2sgmt:
    y_train.append(w2v_model[word].tolist())
y_train = np.array(y_train)
if len(y_train) != len(word2sgmt):
    sys.exit('ERROR: Pre-trained vectors do not contain all words in wordlist !!')
print('number of pre-trained vectors: ', len(w2v_model.vocab))

print('number of words found: ', len(y_train))
print('shape of Y: ', y_train.shape)

print('==========  Save Input and Output...  ==========')
np.save("x_train", x_train)
np.save("y_train", y_train)

print('==========  Build model...  ==========')
morph_seg = [Input(shape=(None,), dtype='int32') for _ in range(number_of_segmentation)]
morph_embedding = Embedding(input_dim=len(morphs) + 1, output_dim=int(dim / 4), mask_zero=True, name="embeddding")

embed_seg = [morph_embedding(x) for x in morph_seg]

bi_lstm = Bidirectional(LSTM(dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=False), merge_mode='concat')

encoded_seg = [bi_lstm(x) for x in embed_seg]

concat_vector = concatenate(encoded_seg, axis=-1)
merge_vector = Reshape((number_of_segmentation, (2 * dim)))(concat_vector)

masked_vector = Masking()(merge_vector)

seq_output = TimeDistributed(Dense(dim))(masked_vector)

attention_1 = TimeDistributed(Dense(units=dim, activation='tanh', use_bias=False))(seq_output)

attention_2 = TimeDistributed(Dense(units=1, activity_regularizer=regularizers.l1(0.01), use_bias=False))(attention_1)


def attn_merge(inputs, mask):
    vectors = inputs[0]
    logits = inputs[1]
    # Flatten the logits and take a softmax
    logits = K.squeeze(logits, axis=2)
    pre_softmax = K.switch(mask[0], logits, -numpy.inf)
    weights = K.expand_dims(K.softmax(pre_softmax))
    return K.sum(vectors * weights, axis=1)


def attn_merge_shape(input_shapes):
    return input_shapes[0][0], input_shapes[0][2]

attn = Lambda(attn_merge, output_shape=attn_merge_shape)
attn.supports_masking = True
attn.compute_mask = lambda inputs, mask: None
content_flat = attn([seq_output, attention_2])

def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)

model = tf.keras.Model(inputs=morph_seg, outputs=content_flat)

model.compile(loss=cosine_proximity, optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=number_of_epoch)

print('==========  Save model weights...  ==========')
model.save_weights(output_file)
print("Model saved in path: %s" % output_file)