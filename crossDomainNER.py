import keras.models
from bs4 import BeautifulSoup
import emoji
import ttp
import re
import string
import nltk
from sklearn.model_selection import train_test_split
import codecs
from tqdm import tqdm
nltk.download('punkt')
punc = string.punctuation
from future.utils import iteritems


def preprocess(text):
    text = ttp.remove_emoji(text)
    text = ttp.remove_emoticon(text)
    text = ttp.remove_user_handle(text)
    text = ttp.remove_hashtag_and_word(text)
    # re.sub(r"http\S+", "", line)

    text = re.sub(r"http\S+", "", text)
    #text = ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())
    text = text.replace("[^a-zA-Z0-9 ']", "")
    text = re.sub(r'[^a-zA-Z0-9 \'ğüşöıçİĞÜŞÖÇ]', ' ', text)
    text = text.lower()
    #alphanumeric_filter = filter(str.isalnum, text)
    #text = "".join(alphanumeric_filter)

    return text


data = "/home/someone/Desktop/cmpe58t/TwitterDS-2/TwitterDS-2/TwitterData_2_Tagged_PLO.txt"
tags = []
words=[]
taggedSentences2=[]
f = open("TwitterData_2_Tagged_PLO.txt", "r")
lines= f.readlines()
for line in lines:
    soup = BeautifulSoup(line,"html.parser")
    sentence=[]
    if len(soup.findAll(attrs={'type': ['PERSON', 'LOCATION', 'ORGANIZATION']})) == 0:

        sentences =nltk.sent_tokenize(line)
        for sent in sentences:
            sent = preprocess(sent)
            tokens = nltk.word_tokenize(sent)

            for token in tokens:
                words.append(token)
                tags.append("O")
                taggedToken = (token, "O")
                sentence.append(taggedToken)

    else:
        soupText = soup.findAll(attrs={'type': ['PERSON', 'LOCATION', 'ORGANIZATION']})
        tag_temp=[]
        tag_text=[]
        for s in soupText:
            ys = s.text.lower().split()
            for y in ys:
                tag_text.append(y)
                tag_temp.append(s.attrs['type'])

        sentences =nltk.sent_tokenize(soup.get_text())
        for sent in sentences:
            sent = preprocess(sent)
            tokens = nltk.word_tokenize(sent)
            for token in tokens:
                t=0
                if token in tag_text:
                    if len(tags) > 0 and ((tags[-1][2:8] == 'PERSON' or tags[-1][2:8] == 'LOCATION' or tags[-1][2:8] == 'ORGANIZATION')):
                        tag = "I-" + tag_temp[t]
                    else:
                        tag = "B-" + tag_temp[t]
                    words.append(token)
                    tags.append(tag)
                    taggedToken = (token, tag)
                    sentence.append(taggedToken)
                    t+=1
                else:
                    words.append(token)
                    tags.append("O")
                    taggedToken = (token, "O")
                    sentence.append(taggedToken)

    taggedSentences2.append(sentence)
wordsSet_2 = list(set(words))
n_words_2 = len(wordsSet_2)

tagsSet_2 = []
for tag in set(tags):
    if tag is None or isinstance(tag, float):
        tagsSet_2.append('unk')
    else:
        tagsSet_2.append(tag)

n_tags_2 = len(tagsSet_2)

word2idx_2 = {w: i for i, w in enumerate(wordsSet_2)}
tag2idx_2 = {t: i for i, t in enumerate(tagsSet_2)}
idx2tag_2 = {v: k for k, v in iteritems(tag2idx_2)}


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import keras as k
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import optimizers
from future.utils import iteritems
import matplotlib.pyplot as plt
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn_crfsuite.metrics import flat_classification_report
from numpy import mean
from numpy import std
import tensorflow as tf



### model1
def utilData(path):
    datal = open(path).readlines()
    taglist=["\"I-PER\",","\"I-LOC\",","\"I-ORG\",","\"B-PER\",","\"B-LOC\",","\"B-ORG\",","\"O\","]
    sentence=[]
    taggedSentences=[]
    words = []
    tags = []
    for datax in datal:
        data=datax.split()
        if len(data)==0:
            taggedSentences.append(sentence)
            sentence=[]
        else:
            a=0
            while a< len(data):
                token=data[a]
                if token in taglist:
                    tag=token[1:-2]
                    a+=1
                    token=data[1]
                    words.append(token)
                    tags.append(tag)
                    taggedToken =(token,tag)
                    sentence.append(taggedToken)
                else:
                    a+= 1

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

n_words, n_tags, word2idx, tag2idx, idx2tag, taggedSentences = utilData("/home/someone/Desktop/cmpe58t/dataset NER/datasets/allData.conllu")

maxlen = max([len(s) for s in taggedSentences])

X = [[word2idx[w[0]] for w in s] for s in taggedSentences]
X = pad_sequences(maxlen=maxlen, sequences=X, padding="post",value=n_words - 1)

y = [[tag2idx[w[1]] for w in s] for s in taggedSentences]
y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

#X_train= X[0:25513]
#X_dev=X[25514:28468]
X_train = X[0:28468]
X_test=X[28469:]

#Y_train= y[0:25513]
#Y_dev=y[25514:28468]
Y_train = X[0:28468]
Y_test=y[28469:]

#maxlen2 = max([len(s) for s in taggedSentences])

X2 = [[word2idx_2[w[0]] for w in s] for s in taggedSentences2]
X2 = pad_sequences(maxlen=maxlen, sequences=X2, padding="post",value=n_words_2 - 1)

y2 = [[tag2idx_2[w[1]] for w in s] for s in taggedSentences2]
y2 = pad_sequences(maxlen=maxlen, sequences=y2, padding="post", value=tag2idx_2["O"])
y2 = [to_categorical(i, num_classes=n_tags_2) for i in y2]

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

embed_dim = 200 #300
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('/home/someone/Desktop/cmpe58t/dataset NER/TweetNER/TweetNER/bounweb+tweetscorpus_twitterprocessed_vectors_lowercase_w5_dim200_fixed.txt', encoding='utf-8')
#f = codecs.open('/home/someone/Desktop/cmpe58t/cc.tr.300.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

words_not_found = []
nb_words = len(word2idx_2)
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word2idx_2.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


input = Input(shape=(maxlen,), dtype='int32')
model = Embedding(input_dim=n_words, output_dim=embed_dim, input_length=maxlen, trainable=False)(input)

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

history = model.fit(X_train, np.array(X_test), batch_size=256, epochs=40, verbose=1)

def pred2label(pred, idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out
test_pred = model.predict(X_test, verbose=1)
pred_labels = pred2label(test_pred, idx2tag)
test_labels = pred2label(Y_test, idx2tag)


print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
print(report)

def transfer_model(model, trainX, trainy, testX, testy, n_fixed, n_repeats):
    scores = list()
    for _ in range(n_repeats):

        temp_model = model

        for i in range(n_fixed):
            temp_model.layers[i].trainable = False

        crf = CRF(n_tags_2)
        out = crf(temp_model.layers[-2].output)
        model2 = Model(temp_model.inputs, out)

        model2.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
        model2.fit(trainX, np.array(trainy), epochs=1, batch_size=32)
        model2.summary()

        test_pred = model.predict(testX, verbose=1)
        pred_labels = pred2label(test_pred, idx2tag_2)
        test_labels = pred2label(testy, idx2tag_2)
        print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

        scores.append(f1_score(test_labels, pred_labels))
        print("fixed layer no: ", n_fixed)
        report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
        print(report)

    return scores

n_repeats= 1
n_fixed = 4
for i in range(n_fixed):
    scores = transfer_model(model,X_train2, Y_train2, X_test2, Y_test2, i, n_repeats)
    #print('Transfer (fixed=%d) %.3f (%.3f)' % (i, mean(scores), std(scores)))


