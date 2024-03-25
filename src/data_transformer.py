import pandas as pd
import numpy as np

from scipy.sparse import hstack

import torch
from torch.nn import functional as F

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


class ohEncoder:
    def fit(self, df, features):
        self.features = features
        self.encoders = {}
        for f in features:
            self.encoders[f] = OneHotEncoder()
            self.encoders[f].fit(df.loc[:, f].values.reshape(-1, 1))

    def transform(self, df):
        X = []
        for f in self.features:
            X.append(self.encoders[f].transform(df.loc[:, f].values.reshape(-1, 1)))
        return hstack(X)


class ohTextEncoder:
    # bag of words
    def fit(self, df, features):
        self.vectorizer = {}
        email_domains = []
        for f in features:
            df[f] = df[f].fillna('n/a.')
            email_domains.append(df[f].str.split('.').apply(lambda x: ' '.join(x)))
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(np.hstack(email_domains))
        self.features = features

    def transform_data(self, df):
        for f in self.features:
            df[f] = df[f].fillna('n/a.')
            df[f] = df[f].str.split('.').apply(lambda x: ' '.join(x))
        return df

    def transform(self, df):
        X = []
        for f in self.features:
            X.append(self.vectorizer.transform(df[f]))
        return X[0] + X[1]


class dataCleaner:
    def __init__(self):
        # gruops of features for different nan filling
        self.nanfiller = {0: [], -99: [], 'median': [], 'mean': [], 'mode': []}
    def fit (self, df, features):
        # unique values in train
        self.feature_values = {}
        for k in []:
            if not k in ['D', 'C', 'V', 'dist', 'transaction']:
                for f in self.feature_values[k]:
                    self.feature_values[f] = df.loc[:, f].value_counts().index

        # cat. str features
        self.nanfiller['n/a'] = features['M'] + features['other_str']

        # cat int features
        self.nanfiller[-99] = features['cat_int'] + features['cat_int_oh']
        self.nan_mask = []

        # num features
        self.nanfiller['mode'] = features['D'] + features['V'] + features['C']
        self.nanfiller['median'] = features['dist']
        
        # mask ffeature names
        self.mask_features = []
        for fill_mode, columns in self.nanfiller.items():
            for f in columns:
                nan_index = df.index[df[f].isna()]
                if len(nan_index) > 0:
                    if fill_mode in [0, 'mode', 'median']:     
                        f_name = f + '_mask'
                        self.mask_features.append(f_name)
                else: 
                    self.nanfiller[fill_mode].remove(f)

    def transform(self, df):
        # new values
        for f in self.feature_values:
            index = df[~df[f].isin(self.feature_values[f])].index
            df.loc[index, f] = np.nan
        # nan filling
        for fill_mode, columns in self.nanfiller.items():
            for f in columns:
                nan_index = df.index[df[f].isna()]

                # masking filling values
                if fill_mode in [0, 'mode', 'median']:
                    f_name = f + '_mask'
                    df.loc[:, f_name] = 0
                    df.loc[nan_index, f_name] = 1
                # nan filling
                if fill_mode in [0, -99, 'n/a']:
                    df.loc[:, f].fillna(fill_mode, inplace=True)
                else:
                    fill_value = df.loc[:, f].agg(fill_mode)
                    if isinstance(fill_value, pd.core.series.Series):
                        fill_value = fill_value.values[0]
                    df.loc[:, f].fillna(fill_value, inplace=True)
        return df


class Scaler:
    def _log_transform(self, x, f):
        return np.log1p(x - self.min_val[f])

    def fit(self, df, features):
        self.scaler = {}
        self.min_val = {}
        for f in features:
            #self.scaler[f] = StandardScaler()
            #self.scaler[f] = MinMaxScaler()
            #self.scaler[f] = Normalizer()
            self.scaler[f] = Pipeline([('scaler1', StandardScaler()), ('scaler2', StandardScaler())])
            x = df.loc[:, f].dropna().values
            self.min_val[f] = min(0, min(x))
            self.scaler[f].fit(self._log_transform(x, f).reshape(-1, 1))
        self.features = features

    def transform(self, df):
        for f in self.features:
            df.loc[:, f] = self.scaler[f].transform(self._log_transform(df.loc[:, f].values, f).reshape(-1, 1))
            #df.loc[:, f] = -1 + 2*df.loc[:, f]
        return df


class DataTransformer():
    def __init__(self, transactions):
        """
        """
        # gruops of features for different nan filling
        self.nanfiller = {0: [], -99: [], 'median': [], 'mean': [], 'mode': []}

        self.features = {}
        # cat. str features
        self.features['M'] = [f for f in transactions.columns if f[0]=='M']
        self.features['email_domains'] = ['P_emaildomain', 'R_emaildomain']
        self.features['other_str'] = ['card4', 'card6', 'ProductCD']

        # cat int features
        self.features['cat_int'] = ['card1', 'card2', 'card3', 'addr1', 'addr2', 'card5']
        self.features['cat_int'] += self.features['other_str']
        self.features['cat_int_oh'] = ['card5']

        # num features
        self.features['D'] = [f for f in transactions.columns if f[0]=='D']
        self.features['C'] = [f for f in transactions.columns if f[0]=='C']
        self.features['V'] = [f for f in transactions.columns if f[0]=='V']
        self.features['dist'] = ['dist1', 'dist2']
        self.features['transaction'] = ['TransactionAmt', 'TransactionDT']
        
        self.features_groups = {}
        
        self.data_cleaner = dataCleaner()
        self.oh_text_encoder = ohTextEncoder()
        self.scaler = Scaler()
        
    def fit(self, df):
        transactions = df.copy()
        
        self.data_cleaner.fit(transactions, self.features)
        transactions = self.data_cleaner.transform(transactions)

        self.features_groups['num'] = []
        for k in ['D', 'C', 'V', 'dist', 'transaction']:
            self.features_groups['num'] += self.features[k]
        
        self.features_groups['oh'] = self.features['M'] + self.features['other_str'] + self.features['cat_int_oh']
        self.features_groups['email_domains'] = self.features['email_domains']

        self.features_groups['train_features'] = self.features_groups['num'] + self.data_cleaner.mask_features
        
        self.scaler.fit(transactions, self.features_groups['num'])
        transactions = self.scaler.transform(transactions)

        self.oh_text_encoder.fit(transactions, self.features_groups['email_domains'])

    def transform_data(self, df):
        transactions = df.copy()
        transactions = self.data_cleaner.transform(transactions)
        transactions = self.scaler.transform(transactions)
        transactions = self.oh_text_encoder.transform_data(transactions)
        return transactions
    
    def get_batch(self, transactions, mode='train'):
        X = transactions.loc[:, self.features_groups['train_features']].fillna(0).values
        X = torch.tensor(X)
        if mode != 'inference':
            Y = transactions.loc[:, 'isFraud'].values
            Y = torch.tensor(Y)
        else:
            Y = None
        return X, Y

