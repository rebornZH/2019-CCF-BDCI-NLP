#! -*- coding:utf-8 -*-

import re
import gc
import codecs
import random
import warnings
import pandas as pd
from sklearn.metrics import f1_score
from nltk.metrics.distance import jaccard_distance
from sklearn.model_selection import StratifiedKFold
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.initializers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

maxlen = 512
config_path = 'F://workplace/BERT/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F://workplace/BERT/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'F://workplace/BERT/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

sub_dir = './result/bert_end2end_classfication_sub.csv'
train_a_dir = './data/a/Train_Data.csv'
test_a_dir = './data/a/Test_Data.csv'
train_b_dir = './data/b/round2_train.csv'
test_b_dir = './data/b/round2_test.csv'


def data_preprocess(x, s):
    data = []
    text_list = re.split("[。！!；;？]", x[s])
    entity_list = x['entity'].split(';')
    for text in text_list:
        for entity in entity_list:
            if entity != '' and entity in text:
                data.append(text)
                break
    if len(data) != 0:
        data = '。'.join(data)
    else:
        data = x[s]
    return data


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def seq_padding(X, padding):
    return np.array([np.concatenate([x, [padding] * (maxlen - len(x))]) if len(x) < maxlen else x[:maxlen] for x in X])


class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                x1, x2 = tokenizer.encode(first=d[0] + '_' + d[1] + '_' + d[2])
                y = d[3]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1, 0)
                    X2 = seq_padding(X2, 0)
                    Y = np.array(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    x = Dense(nclass, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
    print(model.summary())
    return model


def run_cv(nfold, train_list, test_list):
    train_preds = np.zeros((len(train_list), 1))
    test_preds = np.zeros((len(test_list), 1))
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=1017)
    for i, (train_index, valid_index) in enumerate(skf.split(train_list, train_list[:, 3])):
        print('第%s折开始训练' % (i + 1))

        X_train, X_valid, = train_list[train_index, :], train_list[valid_index, :]
        train_D = data_generator(X_train, batch_size=4, shuffle=True)
        valid_D = data_generator(X_valid, batch_size=4, shuffle=True)
        test_D = data_generator(test_list, batch_size=4, shuffle=False)

        K.clear_session()
        model = build_bert(1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')
        plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('./model/bert_end2end_classfication_fold_' + str(i + 1) + '.h5', monitor='val_loss',
                                     verbose=2, save_best_only=True, mode='min', save_weights_only=True)
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=10,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, checkpoint],
        )
        model.save('./model/bert_end2end_classfication_fold_' + str(i + 1) + '.h5')
        train_preds[valid_index] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_preds += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1) / nfold

        del model
        gc.collect()

    return train_preds, test_preds

train_a = pd.read_csv(train_a_dir, index_col=None)
train_b = pd.read_csv(train_b_dir, index_col=None)
train = pd.concat([train_a, train_b], axis=0, ignore_index=True)
test = pd.read_csv(test_b_dir, index_col=None)

print('train', train.shape)
print('test', test.shape)
print(train.columns)
print(test.columns)

print('train content')
train.fillna('', inplace=True)
train['title'] = train.apply(lambda x: data_preprocess(x, 'title'), axis=1)
train['text'] = train.apply(lambda x: data_preprocess(x, 'text'), axis=1)
train['content'] = train.apply(lambda x: x['title'] + '_' + x['text'], axis=1)

print('test content')
test.fillna('', inplace=True)
test['title'] = test.apply(lambda x: data_preprocess(x, 'title'), axis=1)
test['text'] = test.apply(lambda x: data_preprocess(x, 'text'), axis=1)
test['content'] = test.apply(lambda x: x['title'] + '_' + x['text'], axis=1)

print('token_dict')
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

print('tokenizer')
tokenizer = OurTokenizer(token_dict)

print('train_list')
train_id = []
train_list = []
for i in train.index:
    data_row = train.iloc[i]
    entitys = data_row['entity'].split(';')
    for entity in entitys:
        if len(entity) != 0:
            train_id.append(data_row['id'])
            if entity in data_row['key_entity'].split(';'):
                train_list.append((entity, data_row['entity'], data_row['content'], 1))
            else:
                train_list.append((entity, data_row['entity'], data_row['content'], 0))
train_list = np.array(train_list)

print('test_list')
test_id = []
test_list = []
for i in test.index:
    data_row = test.iloc[i]
    entitys = data_row['entity'].split(';')
    for entity in entitys:
        if len(entity) != 0:
            test_id.append(data_row['id'])
            test_list.append((entity, data_row['entity'], data_row['content'], 0))
test_list = np.array(test_list)

print('start train')
train_preds, test_preds = run_cv(5, train_list, test_list)

train_preds = train_preds[:, 0]
test_preds = test_preds[:, 0]
test_pre_result = [1 if test_pred > 0.5 else 0 for test_pred in test_preds]

id_list = []
negative_list = []
key_entity_list = []
tmp = pd.DataFrame({'id': test_id, 'entity': test_list[:, 0], 'pre_result': test_pre_result})
for id in tmp['id'].unique():
    df = tmp[(tmp['id'] == id) & (tmp['pre_result'] == 1)]
    entitys = df['entity'].tolist()
    if len(entitys) > 0:
        id_list.append(id)
        negative_list.append(1)
        key_entity_list.append(';'.join(entitys))
    else:
        id_list.append(id)
        negative_list.append(0)
        key_entity_list.append('')

sub = pd.DataFrame({'id': id_list, 'negative': negative_list, 'key_entity': key_entity_list})
sub = pd.merge(test[['id']], sub, on='id', how='left')
sub['negative'] = sub['negative'].fillna(0).astype(int)
sub['key_entity'] = sub['key_entity'].fillna('')
print(sub['negative'].value_counts())
sub.to_csv(sub_dir, index=None)
