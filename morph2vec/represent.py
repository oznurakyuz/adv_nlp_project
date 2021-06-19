import codecs

import keras.backend as K
import numpy
from keras import regularizers, Model
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Masking
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional

number_of_segmentation = 10

print('===================================  Prepare data...  ==============================================')
print('')

word2sgmt = {}
word2segmentations = {}
seq = []
morphs = []
dim = 300
f = codecs.open('../data_resource/structured_news_10segment.txt', encoding='utf-8')
for line in f:
    line = line.rstrip('\n')
    word, sgmnts = line.split(':')
    sgmt = sgmnts.split('+')
    word2segmentations[word] = list(s for s in sgmt)
    sgmt = list(s.split('-') for s in sgmt)
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

print('')
print('===================================  Build model...  ===============================================')
print('')

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
    pre_softmax = K.switch(mask[0], logits, -numpy.inf)
    weights = K.expand_dims(K.softmax(pre_softmax))
    return K.sum(vectors * weights, axis=1)


def attn_merge_shape(input_shapes):
    return (input_shapes[0][0], input_shapes[0][2])


attn = Lambda(attn_merge, output_shape=attn_merge_shape)
attn.supports_masking = True
attn.compute_mask = lambda inputs, mask: None
content_flat = attn([seq_output, attention_2])

model = Model(inputs=morphs_inp, outputs=content_flat)
model.load_weights("../output/model_news_15062021.h5")

print(model.summary())

m_w = model.get_layer("embedding").get_weights()
print(len(m_w[0]))

m_vectors = {}
for i in range(len(m_w[0])):
    if not indices_morph[i] == '###':
        m_vectors[indices_morph[i]] = m_w[0][i]

import pickle

with open('../output/vectors_news_15062021.vec', 'wb') as fp:
    pickle.dump(m_vectors, fp, protocol=pickle.HIGHEST_PROTOCOL)
