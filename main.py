import pandas as pd
import numpy as np 

data = pd.read_csv("./data/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words)

tags = list(data["Tag"].values)
tags_2 = ["START"] + tags[:-1]

o1_tags = list(set(tags))
n_o1_tags = len(o1_tags)
print(n_o1_tags)

o2_tags = list(set([(tags_2[i]+"/"+tags[i]) for i in range(len(tags))]))
n_o2_tags = len(o2_tags)
print(n_o2_tags)

alo = [(tags_2[i]+"/"+tags[i]) for i in range(len(tags))]
data["Tag_2"] = alo
data.to_csv("./data/ner.csv")

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t1, t2) for w, p, t1, t2 in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist(),
                                                           s["Tag_2"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences

max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
o1_tag2idx = {t1: i for i, t1 in enumerate(o1_tags)}
o2_tag2idx = {t2: i for i, t2 in enumerate(o2_tags)}


from keras.preprocessing.sequence import pad_sequences

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)

o1_y = [[o1_tag2idx[w[2]] for w in s] for s in sentences]
o2_y = [[o2_tag2idx[w[3]] for w in s] for s in sentences]
o1_y = pad_sequences(maxlen=max_len, sequences=o1_y, padding="post", value=o1_tag2idx["O"])
o2_y = pad_sequences(maxlen=max_len, sequences=o2_y, padding="post", value=o2_tag2idx["O/O"])

from keras.utils import to_categorical

o1_y = [to_categorical(i, num_classes=n_o1_tags) for i in o1_y]
o2_y = [to_categorical(i, num_classes=n_o2_tags) for i in o2_y]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, o1_y, test_size=0.1)

from keras.models import Model 
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input
from keras_contrib.layers import CRF

inp = Input(shape=(max_len,))
model = Embedding(n_words + 1,
                    output_dim=20, 
                    input_length=max_len, 
                    mask_zero=True)(inp)
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)

crf = CRF(n_o1_tags)
out = crf(model)

model = Model(inp, out)
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])

model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

from sklearn.metrics import f1_score

test_pred = model.predict(X_test, verbose=1)
o1_idx2tag = {i: w for w, i in o1_tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(o1_idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)

print(f1_score(test_labels, pred_labels))