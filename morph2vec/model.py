import codecs
import os
import sys

import keras.backend as K
import numpy as np
from gensim.models import fasttext
from keras import regularizers, Model
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Masking
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence

number_of_segmentation = 10  # args.segNo
batch_size = 32  # args.batch
number_of_epoch = 50  # 25  # args.epoch
dim = 300  # args.dim

gensim_model = "../vectors/cc.tr.300.bin"  # args.wordVector
training_file = "../data_resource/structured_news_10segment.txt"  # args.input
output_file = "../output/model_news_15062021.h5"  # args.output

if not os.path.exists("../output/"):
    os.makedirs("../output/")

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

x_train = [[] for i in range(number_of_segmentation)]
for word in word2sgmt:
    for i in range(len(word2sgmt[word])):
        x_train[i].append([morph_indices[c] for c in word2sgmt[word][i]])

for i in range(number_of_segmentation):
    x_train[i] = np.array(x_train[i], dtype=object)

for i in range(len(x_train)):
    x_train[i] = sequence.pad_sequences(x_train[i], maxlen=timesteps_max_len)

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
morphs_inp = [Input(shape=(None,), dtype='int32') for _ in range(number_of_segmentation)]
inputs = Concatenate()(morphs_inp)
embeds = Embedding(input_dim=len(set(morphs)) + 1, output_dim=int(dim / 4), mask_zero=True, name="embedding")(inputs)
bilstm = Bidirectional(LSTM(dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), name="BiLSTM",
                       merge_mode='concat')(embeds)

x1 = Masking()(bilstm)
seq_output = TimeDistributed(Dense(dim))(x1)
attention_1 = TimeDistributed(Dense(units=dim, activation='tanh', use_bias=False))(seq_output)
attention_2 = TimeDistributed(Dense(units=1, activity_regularizer=regularizers.l1(0.01), use_bias=False))(attention_1)


def attn_merge(inputs, mask):
    vectors = inputs[0]
    logits = inputs[1]
    # Flatten the logits and take a softmax
    logits = K.squeeze(logits, axis=2)
    pre_softmax = K.switch(mask[0], logits, -np.inf)
    weights = K.expand_dims(K.softmax(pre_softmax))
    return K.sum(vectors * weights, axis=1)


def attn_merge_shape(input_shapes):
    return input_shapes[0][0], input_shapes[0][2]


attn = Lambda(attn_merge, output_shape=attn_merge_shape)
attn.supports_masking = True
attn.compute_mask = lambda inputs, mask: None
content_flat = attn([seq_output, attention_2])

model = Model(inputs=morphs_inp, outputs=content_flat)
print(model.summary())


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


model.compile(loss=cosine_proximity, optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=number_of_epoch)

print('==========  Save model weights...  ==========')
model.save_weights(output_file)
print("Model saved in path: %s" % output_file)
