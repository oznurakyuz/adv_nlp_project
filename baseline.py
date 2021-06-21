import string

import keras as k
import matplotlib.pyplot as plt
import numpy as np
from future.utils import iteritems
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from seqeval.metrics import f1_score
from sklearn_crfsuite.metrics import flat_classification_report

punc = string.punctuation


def utilData(path):
    datal = open(path).readlines()
    taglist = ["\"I-PER\",", "\"I-LOC\",", "\"I-ORG\",", "\"B-PER\",", "\"B-LOC\",", "\"B-ORG\","]
    sentence = []
    taggedSentences = []
    words = []
    tags = []
    for datax in datal:
        data = datax.split()
        if len(data) == 0:
            taggedSentences.append(sentence)
            sentence = []
        else:
            a = 0
            while a < len(data) and data[1] not in punc:
                token = data[a]
                if token == '"O",':
                    tag = token[1:-2]
                    a += 1
                    token = data[1].lower()
                    words.append(token)
                    tags.append(tag)
                    taggedToken = (token, tag)
                    sentence.append(taggedToken)
                elif token in taglist:
                    if len(tags) > 0 and ((tags[-1][2:5] == 'PER' or tags[-1][2:5] == 'LOC' or tags[-1][2:5] == 'ORG')):
                        tag = token[1:-2]
                    else:
                        tag = "B-" + token[3:-2]
                    a += 1
                    token = data[1].lower()
                    words.append(token)
                    tags.append(tag)
                    taggedToken = (token, tag)
                    sentence.append(taggedToken)
                else:
                    a += 1

    print(taggedSentences[1:3])
    wordsSet = list(set(words))
    n_words = len(wordsSet)

    tagsSet = []
    for tag in set(tags):
        if tag is None or isinstance(tag, float):
            tagsSet.append('unk')
        else:
            tagsSet.append(tag)

    n_tags = len(tagsSet)

    word2idx = {w: i for i, w in enumerate(wordsSet)}
    tag2idx = {t: i for i, t in enumerate(tagsSet)}
    idx2tag = {v: k for k, v in iteritems(tag2idx)}
    return n_words, n_tags, word2idx, tag2idx, idx2tag, taggedSentences


n_words, n_tags, word2idx, tag2idx, idx2tag, taggedSentences = utilData(
    "/home/someone/Desktop/cmpe58t/dataset NER/datasets/allData.conllu")
# n_words, n_tags, word2idx, tag2idx, idx2tag, taggedSentences = utilData("/home/someone/Desktop/cmpe58t/dataset NER/datasets/turkish-ner-train.conllu")
# n_words, n_tags, word2idx, tag2idx, idx2tag, taggedSentences = utilData("/home/someone/Desktop/cmpe58t/dataset NER/datasets/turkish-ner-test.conllu")


maxlen = max([len(s) for s in taggedSentences])

X = [[word2idx[w[0]] for w in s] for s in taggedSentences]
X = pad_sequences(maxlen=maxlen, sequences=X, padding="post", value=n_words - 1)

x_split1 = X[0:10]
x_split2 = X[:-100]

y = [[tag2idx[w[1]] for w in s] for s in taggedSentences]
y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_train = X[0:25513]
X_dev = X[25514:28468]
X_test = X[28469:]

y_train = y[0:25513]
y_dev = y[25514:28468]
y_test = y[28469:]

print("maxlen", maxlen)
input = Input(shape=(maxlen,))
word_embedding_size = 100

model = Embedding(input_dim=n_words, output_dim=word_embedding_size, input_length=maxlen)(input)

model = Bidirectional(LSTM(units=64,
                           return_sequences=True,
                           dropout=0.5,
                           recurrent_dropout=0.5,
                           kernel_initializer=k.initializers.he_normal()))(model)

model = Bidirectional(LSTM(units=64,
                           return_sequences=True,
                           dropout=0.5,
                           recurrent_dropout=0.5,
                           kernel_initializer=k.initializers.he_normal()))(model)

model = TimeDistributed(Dense(100, activation="relu"))(model)

crf = CRF(n_tags)

out = crf(model)
model = Model(input, out)

adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
model.summary()

filepath = "ner-bi-lstm-td-model-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=1, validation_data=(X_dev, np.array(y_dev)),
                    verbose=1)

plt.style.use('ggplot')


def plot_history(history):
    accuracy = history.history['crf_viterbi_accuracy']
    val_accuracy = history.history['val_crf_viterbi_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, 'b', label='Training acc')
    plt.plot(x, val_accuracy, 'r', label='Val acc')
    plt.title('Training and val accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Val loss')
    plt.title('Training and val loss')
    plt.legend()
    plt.savefig('dev_epoch40_plot.png')


plot_history(history)


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out


test_pred = model.predict(X_test, verbose=1)
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
print(report)
