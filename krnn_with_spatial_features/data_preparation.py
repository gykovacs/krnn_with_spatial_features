# %% Data preparation

import copy
import numpy as np
import os
import pandas as pd
import sys

from scipy.io import arff
from scipy.spatial import distance_matrix
from scipy.stats import ttest_ind

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# %%

base_path = '/home/gykovacs/workspaces/imbalanced-knn/data/'


# %% Encoding functions


def encode_column_onehot(column):
    lencoder = LabelEncoder().fit(column)
    lencoded = lencoder.transform(column)
    ohencoder = OneHotEncoder(sparse=False).fit(lencoded.reshape(-1, 1))
    ohencoded = ohencoder.transform(lencoded.reshape(-1, 1))

    return ohencoded


def encode_column_median(column, missing_values):
    column = copy.deepcopy(column)
    if np.sum([np.in1d(column, missing_values)]) > 0:
        column[np.in1d(column, missing_values)] = np.median(
            column[~np.in1d(column, missing_values)].astype(float))
    column = column.astype(float)
    return column.values


def encode_features(data, target='target', encoding_threshold=4, missing_values=['?', None, 'None']):
    columns = []
    column_names = []

    for c in data.columns:
        sys.stdout.write('Encoding column %s' % c)
        if not c == target:
            # If the column is not the target variable
            n_values = len(np.unique(data[c]))
            sys.stdout.write(' number of values: %d => ' % n_values)

            if n_values == 1:
                # There is no need for encoding
                sys.stdout.write('no encoding\n')
                continue
            elif n_values < encoding_threshold:
                # Applying one-hot encoding
                sys.stdout.write('one-hot encoding\n')
                ohencoded = encode_column_onehot(data[c])
                for i in range(ohencoded.shape[1]):
                    columns.append(ohencoded[:, i])
                    column_names.append(str(c) + '_onehot_' + str(i))
            else:
                # Applying median encoding
                sys.stdout.write(
                    'no encoding, missing values replaced by medians\n')
                columns.append(encode_column_median(data[c], missing_values))
                column_names.append(c)

        if c == target:
            # In the target column the least frequent value is set to 1, the
            # rest is set to 0
            sys.stdout.write(' target variable => least frequent value is 1\n')
            column = copy.deepcopy(data[c])
            val_counts = data[target].value_counts()
            if val_counts.values[0] < val_counts.values[1]:
                mask = (column == val_counts.index[0])
                column[mask] = 1
                column[~(mask)] = 0
            else:
                mask = (column == val_counts.index[0])
                column[mask] = 0
                column[~(mask)] = 1

            columns.append(column.astype(int).values)
            column_names.append(target)

    return pd.DataFrame(np.vstack(columns).T, columns=column_names)


def read_hiva():
    db = pd.read_csv(os.path.join(base_path, 'hiva',
                                  'hiva_train.data'), sep=' ', header=None)
    del db[db.columns[-1]]
    target = pd.read_csv(os.path.join(
        base_path, 'hiva', 'hiva_train.labels'), header=None)
    db['target'] = target

    return encode_features(db)


def read_hypothyroid():
    db = pd.read_csv(os.path.join(base_path, 'hypothyroid',
                                  'hypothyroid.data.txt'), header=None)
    db.columns = ['target'] + list(db.columns[1:])

    return encode_features(db)


def read_sylva():
    db = pd.read_csv(os.path.join(base_path, 'sylva',
                                  'sylva_train.data'), sep=' ', header=None)
    del db[db.columns[-1]]
    target = pd.read_csv(os.path.join(base_path, 'sylva',
                                      'sylva_train.labels'), header=None)
    db['target'] = target

    return encode_features(db)


def read_pc1():
    data, meta = arff.loadarff(os.path.join(base_path, 'pc1', 'pc1.arff'))
    db = pd.DataFrame(data)
    db.set_value(db['defects'] == b'false', 'defects', False)
    db.set_value(db['defects'] == b'true', 'defects', True)

    db.columns = list(db.columns[:-1]) + ['target']
    return encode_features(db)


def read_cm1():
    data, meta = arff.loadarff(os.path.join(base_path, 'cm1', 'cm1.arff.txt'))
    db = pd.DataFrame(data)
    db.set_value(db['defects'] == b'false', 'defects', False)
    db.set_value(db['defects'] == b'true', 'defects', True)
    db.columns = list(db.columns[:-1]) + ['target']

    return encode_features(db)


def read_kc1():
    data, meta = arff.loadarff(os.path.join(base_path, 'kc1', 'kc1.arff.txt'))
    db = pd.DataFrame(data)
    db.set_value(db['defects'] == b'false', 'defects', False)
    db.set_value(db['defects'] == b'true', 'defects', True)
    db.columns = list(db.columns[:-1]) + ['target']

    return encode_features(db)


def read_spectf():
    db0 = pd.read_csv(os.path.join(base_path, 'spect_f',
                                   'SPECTF.train.txt'), header=None)
    db1 = pd.read_csv(os.path.join(base_path, 'spect_f',
                                   'SPECTF.test.txt'), header=None)
    db = pd.concat([db0, db1])
    db.columns = ['target'] + list(db.columns[1:])

    return encode_features(db)


def read_hepatitis():
    db = pd.read_csv(os.path.join(base_path, 'hepatitis',
                                  'hepatitis.data.txt'), header=None)
    db.columns = ['target'] + list(db.columns[1:])

    return encode_features(db)


def read_vehicle():
    db0 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xaa.dat.txt'), sep=' ', header=None, usecols=range(19))
    db1 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xab.dat.txt'), sep=' ', header=None, usecols=range(19))
    db2 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xac.dat.txt'), sep=' ', header=None, usecols=range(19))
    db3 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xad.dat.txt'), sep=' ', header=None, usecols=range(19))
    db4 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xae.dat.txt'), sep=' ', header=None, usecols=range(19))
    db5 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xaf.dat.txt'), sep=' ', header=None, usecols=range(19))
    db6 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xag.dat.txt'), sep=' ', header=None, usecols=range(19))
    db7 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xah.dat.txt'), sep=' ', header=None, usecols=range(19))
    db8 = pd.read_csv(os.path.join(base_path, 'vehicle',
                                   'xai.dat.txt'), sep=' ', header=None, usecols=range(19))

    db = pd.concat([db0, db1, db2, db3, db4, db5, db6, db7, db8])

    db.columns = list(db.columns[:-1]) + ['target']
    db.set_value(db['target'] != 'van', 'target', 'other')

    return encode_features(db)


def read_ada():
    db = pd.read_csv(os.path.join(base_path, 'ada',
                                  'ada_train.data'), sep=' ', header=None)
    del db[db.columns[-1]]
    target = pd.read_csv(os.path.join(
        base_path, 'ada', 'ada_train.labels'), header=None)
    db['target'] = target

    return encode_features(db)


def read_german():
    db = pd.read_csv(os.path.join(base_path, 'german',
                                  'german.data.txt'), sep=' ', header=None)
    db.columns = list(db.columns[:-1]) + ['target']

    return encode_features(db, encoding_threshold=20)


def read_glass():
    db = pd.read_csv(os.path.join(base_path, 'glass',
                                  'glass.data.txt'), header=None)
    db.columns = list(db.columns[:-1]) + ['target']
    db.set_value(db['target'] != 3, 'target', 0)
    del db[db.columns[0]]

    return encode_features(db)


def read_satimage():
    db0 = pd.read_csv(os.path.join(base_path, 'satimage',
                                   'sat.trn.txt'), sep=' ', header=None)
    db1 = pd.read_csv(os.path.join(base_path, 'satimage',
                                   'sat.tst.txt'), sep=' ', header=None)
    db = pd.concat([db0, db1])
    db.columns = list(db.columns[:-1]) + ['target']
    db.set_value(db['target'] != 4, 'target', 0)

    return encode_features(db)


# %% Utility functions


def statistics(dbs):
    for d in dbs:
        name = d
        size = len(dbs[d])
        attr = len(dbs[d].iloc[0]) - 1

        print(d)

        classes = np.unique(dbs[d]['target'])
        pos = np.sum(dbs[d]['target'] == classes[0])
        neg = np.sum(dbs[d]['target'] == classes[1])

        if neg < pos:
            neg, pos = pos, neg
            classes[0], classes[1] = classes[1], classes[0]

        features = copy.deepcopy(dbs[d].columns[dbs[d].columns != 'target'])
        f = dbs[d][features].as_matrix()
        t = dbs[d]['target'].as_matrix()
        prep = preprocessing.StandardScaler().fit(f)
        f = prep.transform(f)
        pos_vectors = f[t == classes[1]]
        neg_vectors = f[t == classes[0]]

        dm = distance_matrix(pos_vectors, pos_vectors)
        dm.sort(axis=1)
        pd = dm[:, 1]
        pos_dists = np.mean(dm[:, 1])
        pos_std = np.std(dm[:, 1])

        dm = distance_matrix(neg_vectors, neg_vectors)
        dm.sort(axis=1)
        nd = dm[:, 1]
        neg_dists = np.mean(dm[:, 1])
        neg_std = np.std(dm[:, 1])

        p_value = ttest_ind(pd, nd, equal_var=False)

        print('%15s\t%d\t%d\t%26s\t%d:%d (%.2f)\t%.2f:%.2f\t%.2f:%.2f\t%f'
              % (name, size, attr, str(classes), pos, neg, float(pos) / size * 100,
                 pos_dists, neg_dists, pos_std, neg_std, p_value[1]))


# %% Reading in datasets


def read_all_datasets():
    dbs = {}

    # TODO(gykovacs): 1600+ dimensions causes math range error,
    #                 expression should be simplified
    #dbs['hiva']= read_hiva()

    # TODO(gykovacs): missing values, categorical
    dbs['hypothyroid'] = read_hypothyroid()

    # Works, very slow with LOOCV
    # dbs['sylva']= read_sylva()  # Works

    dbs['glass'] = read_glass()  # Works

    dbs['pc1'] = read_pc1()  # Works
    dbs['cm1'] = read_cm1()  # Works
    dbs['kc1'] = read_kc1()  # Works
    dbs['spectf'] = read_spectf()  # Works

    # TODO(gykovacs): missing values should be handled
    dbs['hepatitis'] = read_hepatitis()

    dbs['vehicle'] = read_vehicle()  # Works
    # TODO(laszlzso): why disabled?
    # dbs['ada']= read_ada()  # Works

    # TODO(gykovacs): categorical attributes need to be handled
    #dbs['german']= read_german()

    # dbs['satimage']= read_satimage()  # Works
