
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
from  sklearn_crfsuite.metrics import flat_classification_report
import string
from TurkishStemmer import TurkishStemmer
import codecs
from tqdm import tqdm

punc = string.punctuation

from bs4 import BeautifulSoup
import emoji
import ttp
import re
import string
import nltk
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

nltk.download('punkt')
punc = string.punctuation
from future.utils import iteritems

from nltk.stem import PorterStemmer
#stemmer = PorterStemmer()
stemmer =TurkishStemmer()
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
        text = soup.get_text().lower().split()


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
                temp = stemmer.stem(token)
                t=0
                str=' '
                res = [i for i in tag_text if temp in i]
                res = str.join(res)
                if len(res) > 0:
                    if len(tags) > 0 and ((tags[-1][2:8] == 'PERSON' or tags[-1][2:8] == 'LOCATION' or tags[-1][2:8] == 'ORGANIZATION')):
                        tag = "I-" + tag_temp[t]
                    else:
                        tag = "B-" + tag_temp[t]
                    words.append(res)
                    tags.append(tag)
                    taggedToken = (res, tag)
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


maxlen = max([len(s) for s in taggedSentences2])

X2 = [[word2idx_2[w[0]] for w in s] for s in taggedSentences2]
X2 = pad_sequences(maxlen=maxlen, sequences=X2, padding="post",value=n_words_2 - 1)

y2 = [[tag2idx_2[w[1]] for w in s] for s in taggedSentences2]
y2 = pad_sequences(maxlen=maxlen, sequences=y2, padding="post", value=tag2idx_2["O"])
y2 = [to_categorical(i, num_classes=n_tags_2) for i in y2]

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag_2[p_i])
        out.append(out_i)
    return out


X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_train2, X_dev2, Y_train2, Y_dev2 = train_test_split(X_train2, Y_train2, test_size=0.2, random_state=42)

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
model = Embedding(input_dim=n_words_2, output_dim=embed_dim, input_length=maxlen, trainable=False)(input)

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

crf = CRF(n_tags_2)

out = crf(model)
model = Model(input, out)

adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
model.summary()

history = model.fit(X_train2, np.array(Y_train2), batch_size=256, epochs=40,validation_data=(X_dev2,np.array(Y_dev2)), verbose=1)

#history = model.fit(X_train2, np.array(Y_train2), batch_size=256, epochs=40, verbose=1)

test_pred = model.predict(X_test2, verbose=1)
pred_labels = pred2label(test_pred)
test_labels = pred2label(Y_test2)
f_score= f1_score(test_labels, pred_labels)
print("F1-score: {:.1%}".format(f_score))
report = flat_classification_report(y_pred=pred_labels, y_true=test_labels)
print(report)


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
    plt.savefig('tweet_epoch40_plot.png')

plot_history(history)
