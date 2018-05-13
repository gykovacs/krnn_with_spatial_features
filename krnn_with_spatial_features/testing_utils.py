# %% Testing logic


import collections
import sys

import numpy as np

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit


# %% Calculating measures


def calculate_measures(prob_scores, pred_labels, true_labels, positive_label):
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0

    for i in range(len(pred_labels)):
        if true_labels[i] == positive_label:
            if pred_labels[i] == true_labels[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if pred_labels[i] == true_labels[i]:
                tn = tn + 1
            else:
                fp = fp + 1

    print(tp, tn, fp, fn)

    results = {}
    results['acc'] = (tp + tn)/(tp + tn + fp + fn)
    results['sens'] = tp / (tp + fn)
    results['spec'] = tn / (fp + tn)
    results['ppv'] = tp / (tp + fp) if tp + fp > 0 else 0.0
    results['npv'] = tn / (tn + fn) if tn + fn > 0 else 0.0
    results['bacc'] = (tp/(tp + fn) + tn/(fp + tn))/2.0
    results['f1'] = 2*tp/(2*tp + fp + fn)

    # TODO: check the way AUC is computed, using convex hull may be useful
    results['auc'] = roc_auc_score(true_labels, np.matrix(prob_scores)[:, 1])

    return results


# %% Cross-validation methods


def loocv(X, labels, classifier):

    positive_label = collections.Counter(labels).most_common()[-1][0]
    negative_label = collections.Counter(labels).most_common()[-2][0]

    labels_binary = np.ndarray(shape=(len(labels)), dtype=int)
    labels_binary[labels == positive_label] = 1
    labels_binary[labels == negative_label] = 0
    labels_binary.astype(int)

    prob_scores = []
    true_labels = []
    pred_labels = []

    for i in range(len(X)):
        print(i)
        training_X = np.vstack((X[:i], X[(i+1):]))
        training_labels = np.hstack((labels_binary[:i], labels_binary[(i+1):]))
        test_X = X[i].reshape(1, -1)
        test_label = labels_binary[i]

        prep = preprocessing.StandardScaler().fit(training_X)

        training_X_preproc = prep.transform(training_X)
        test_X_preproc = prep.transform(test_X)

        classifier.fit(training_X_preproc, training_labels)
        prob = classifier.predict_proba(test_X_preproc)[0]

        if prob[0] > prob[1]:
            pred = 0
        else:
            pred = 1

        prob_scores.append(prob)
        pred_labels.append(pred)
        true_labels.append(test_label)

    return calculate_measures(prob_scores, pred_labels, true_labels, 1)


def shufflecv(X, labels, classifier, k):

    positive_label = collections.Counter(labels).most_common()[-1][0]
    negative_label = collections.Counter(labels).most_common()[-2][0]

    labels_binary = np.ndarray(shape=(len(labels)), dtype=int)
    labels_binary[labels == positive_label] = 1
    labels_binary[labels == negative_label] = 0
    labels_binary.astype(int)

    prob_scores = []
    true_labels = []
    pred_labels = []

#    random.seed(1)

    #kf= KFold(n_splits= k)
    kf = ShuffleSplit(n_splits=k, test_size=0.1, random_state=1)

    j = 0
    for train, test in kf.split(X):
        #print('stage: %d' % j)
        j = j + 1
        training_X = X[train]
        training_labels = labels_binary[train]
        test_X = X[test]
        test_labels = labels_binary[test]

        prep = preprocessing.StandardScaler().fit(training_X)
        training_X_preproc = prep.transform(training_X)

        classifier.fit(training_X_preproc, training_labels)

        for t in range(len(test)):
            test_X_preproc = prep.transform(test_X[t].reshape(1, -1))
            prob = classifier.predict_proba(test_X_preproc)[0]
            if prob[0] > prob[1]:
                pred = 0
            else:
                pred = 1

            prob_scores.append(prob)
            pred_labels.append(pred)
            true_labels.append(test_labels[t])

    return calculate_measures(prob_scores, pred_labels, true_labels, 1)

# %% Executing classifiers on datasets


def execute_on_all_datasets(classifier, dbs, method='loocv'):
    results = {}
    for name in dbs:
        print('dataset: %s' % name)

        features = dbs[name].columns[dbs[name].columns != 'target']
        if method == 'loocv':
            results[name] = loocv(dbs[name][features].as_matrix(
            ), dbs[name]['target'].as_matrix(), classifier)
        else:
            results[name] = shufflecv(dbs[name][features].as_matrix(
            ), dbs[name]['target'].as_matrix(), classifier, 20)

        print(results[name])

    return results


# %% Printing results


def print_results(datasets, measures):
    rows = []
    for ds_name in datasets[0]:
        rows.append([])
        for ds in datasets:
            for m in measures:
                sys.stdout.write('%.4f & ' % ds[ds_name][m])
                rows[-1].append(ds[ds_name][m])
        sys.stdout.write('\n')

    for i in range(len(rows[0])):
        values = []
        for j in range(len(rows)):
            values.append(rows[j][i])
        sys.stdout.write('%.4f & ' % np.mean(values))
    sys.stdout.write('\n')


def print_results_2(results):
    for w in results:
        print(w)
        d1 = results[w][0]
        d2 = results[w][1]
        d3 = results[w][2]
        d4 = results[w][3]
        d5 = results[w][4]
        for ds in d1:
            print("%s %s %.4f" % ('None', ds, d1[ds]['auc']))
        print(np.mean([d1[ds]['auc'] for ds in d1]))
        for ds in d2:
            print("%s %s %.4f" % ('R_1', ds, d2[ds]['auc']))
        print(np.mean([d2[ds]['auc'] for ds in d2]))
        for ds in d3:
            print("%s %s %.4f" % ('R_2', ds, d3[ds]['auc']))
        print(np.mean([d3[ds]['auc'] for ds in d3]))
        for ds in d4:
            print("%s %s %.4f" % ('R_3', ds, d4[ds]['auc']))
        print(np.mean([d4[ds]['auc'] for ds in d4]))
        for ds in d5:
            print("%s %s %.4f" % ('R_all', ds, d5[ds]['auc']))
        print(np.mean([d5[ds]['auc'] for ds in d5]))


# %% Aggregated results


def aggregated_results(measure, results):
    xs = []
    ys1 = []
    ys2 = []
    ys3 = []
    ys4 = []
    ys5 = []
    for w in results:
        d1 = results[w][0]
        d2 = results[w][1]
        d3 = results[w][2]
        d4 = results[w][3]
        d5 = results[w][4]
        values1 = []
        values2 = []
        values3 = []
        values4 = []
        values5 = []
        for ds in d1:
            values1.append(d1[ds][measure])
        for ds in d2:
            values2.append(d2[ds][measure])
        for ds in d3:
            values3.append(d3[ds][measure])
        for ds in d4:
            values4.append(d4[ds][measure])
        for ds in d5:
            values5.append(d5[ds][measure])
        xs.append(w)
        ys1.append(np.mean(values1))
        ys2.append(np.mean(values2))
        ys3.append(np.mean(values3))
        ys4.append(np.mean(values4))
        ys5.append(np.mean(values5))

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.04f & ' % xs[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys1[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys2[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys3[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys4[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys5[i])
    sys.stdout.write('\n')

    return xs, ys1, ys2, ys3, ys4, ys5


def aggregated_resultsk(measure, resultsk):
    xs = []
    ys1 = []
    ys2 = []
    ys3 = []
    ys4 = []
    ys5 = []
    for w in resultsk:
        d1 = resultsk[w][0]
        d2 = resultsk[w][1]
        d3 = resultsk[w][2]
        d4 = resultsk[w][3]
        d5 = resultsk[w][4]
        values1 = []
        values2 = []
        values3 = []
        values4 = []
        values5 = []
        for ds in d1:
            values1.append(d1[ds][measure])
        for ds in d2:
            values2.append(d2[ds][measure])
        for ds in d3:
            values3.append(d3[ds][measure])
        for ds in d4:
            values4.append(d4[ds][measure])
        for ds in d5:
            values5.append(d5[ds][measure])
        xs.append(w)
        ys1.append(np.mean(values1))
        ys2.append(np.mean(values2))
        ys3.append(np.mean(values3))
        ys4.append(np.mean(values4))
        ys5.append(np.mean(values5))

    for i in range(len(ys1)):
        sys.stdout.write('%d & ' % xs[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys1[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys2[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys3[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys4[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys5[i])
    sys.stdout.write('\n')

    return xs, ys1, ys2, ys3, ys4, ys5
