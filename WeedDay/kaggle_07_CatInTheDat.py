# kaggle_07_CatInTheDat.py
import pandas as pd
import numpy as np
from sklearn import linear_model, feature_extraction
from sklearn import preprocessing, model_selection


def show_accuracy(x, y):
    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    lr = linear_model.LogisticRegression()
    lr.fit(x_train, y_train)

    print('acc :', lr.score(x_test, y_test))


def show_label_encoder(df, y):
    train = pd.DataFrame()
    enc = preprocessing.LabelEncoder()

    for col in df.columns:
        if df[col].dtype == np.object:
            train[col] = enc.fit_transform(df[col])
        else:
            train[col] = df[col]

    print('shape : {} LabelEncoder'.format(train.shape))
    show_accuracy(train, y)


def show_onehot_encoder(df, y):
    enc = preprocessing.OneHotEncoder()
    train = enc.fit_transform(df)

    print('shape : {} OneHotEncoder'.format(train.shape))
    show_accuracy(train, y)


def show_feature_hasher(df, y):
    df_hash = df.copy()
    for col in df_hash.columns:
        df_hash[col] = df[col].astype('str')

    hashing = feature_extraction.FeatureHasher(input_type='string')
    train = hashing.fit_transform(df_hash.values)

    print('shape : {} FeatureHasher'.format(train.shape))
    show_accuracy(train, y)


df = pd.read_csv('kaggle/cat_train.csv', index_col=0)

y = df.target
x = df.drop(['target'], axis=1)

# show_label_encoder(x, y)
# show_onehot_encoder(x, y)
show_feature_hasher(x, y)

# shape : (300000, 23) LabelEncoder
# acc : 0.6885666666666667

# shape : (300000, 16461) OneHotEncoder
# acc : 0.75855

# shape : (300000, 1048576) FeatureHasher
# acc : 0.75165
