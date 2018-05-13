# %% Testing implementation


import datetime

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from wrappers.wrapper_metacost import MetaCostWrapper
from wrappers.wrapper_smote import SMOTEWrapper
from classifiers.class_knn import kNNSimple
from classifiers.class_enn import ENN
from classifiers.class_pnn import PNN
from classifiers.class_krnn import KRNN
from classifiers.class_ldc_knn_impl import kNNLocalDensityCorrection
from testing_utils import execute_on_all_datasets
from testing_utils import aggregated_results, print_results


# %%


krnn_none = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                              correction=None), 'shufflecv')


# %%


krnn_r1 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r1',
                                                            w=0.1), 'shufflecv')


# %%

krnn_r3 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r3',
                                                            w=0.1), 'shufflecv')


# %%


krnn_r4 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r4',
                                                            w=0.1), 'shufflecv')


# %%


krnn_r4 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r_all',
                                                            w=0.1), 'shufflecv')


# %%


for w in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    krnn_r1 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                correction='r1', w=w))

krnn_none = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                              correction=None), 'shufflecv')


krnn_pure_r1 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                 correction='pure_r1', w=0.0001), 'shufflecv')


# %%


print_results([krnn_none, krnn_pure_r1], [
              'auc', 'acc', 'bacc', 'spec', 'sens'])

    
# %%


krnn_pure_r2 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                 correction='pure_r2', w=0.0001), 'shufflecv')

krnn_none = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                              correction=None), 'shufflecv')

krnn_r1 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r1', w=0.0001), 'shufflecv')

krnn_r2 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r2', w=0.0001), 'shufflecv')

krnn_r1_005 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                correction='r1', w=0.05), 'shufflecv')

krnn_r2_005 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                correction='r2', w=0.05), 'shufflecv')

krnn_r1_ens = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                correction='ens_r1', w=0.05), 'shufflecv')

krnn_r2_ens = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                                correction='ens_r2', w=0.05), 'shufflecv')


# %%


results = {}
for w in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
    print(datetime.datetime.now())
    print(w)
    print(w, 'None')
    a = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                          correction=None, w=w), 'shufflecv')
    print(w, 'r1')
    b = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                          correction='r1', w=w), 'shufflecv')
    print(w, 'r3')
    c = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                          correction='r3', w=w), 'shufflecv')
    print(w, 'r4')
    d = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                          correction='r4', w=w), 'shufflecv')
    print(w, 'r_all')
    e = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                          correction='r_all', w=w), 'shufflecv')

    results[w] = [a, b, c, d, e]


# %%


resultsk = {}
for k in [1, 3, 5, 7]:
    print(k)
    print(k, 'None')
    a = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=k,
                                                          correction=None), 'shufflecv')
    print(k, 'r1')
    b = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=k,
                                                          correction='r1', w=0.2), 'shufflecv')

    print(k, 'r3')
    c = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=k,
                                                          correction='r3', w=0.2), 'shufflecv')
    print(k, 'r4')
    d = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=k,
                                                          correction='r4', w=0.2), 'shufflecv')

    print(k, 'r_all')
    e = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=k,
                                                          correction='r_all', w=0.2), 'shufflecv')
    resultsk[k] = [a, b, c, d, e]


# %%


x, auc_1, auc_2, auc_3, auc_4, auc_5 = aggregated_results('auc')
x, f1_1, f1_2, f1_3, f1_4, f1_5 = aggregated_results('f1')
x, bacc_1, bacc_2, bacc_3, bacc_4, bacc_5 = aggregated_results('bacc')


# %%


print_results([krnn_none, results[1.0][0], results[0.01][1],
               krnn_pure_r1, krnn_pure_r2, krnn_r1_ens, krnn_r2_ens], ['auc'])


# %%


x, auc_1, auc_2 = aggregated_results('auc')
x, f1_1, f1_2 = aggregated_results('f1')
x, bacc_1, bacc_2 = aggregated_results('bacc')

plt.figure(figsize=(10, 3))
plt.plot(x, auc_1, label='AUC R1', c='r', ls='-')
plt.plot(x, auc_2, label='AUC R2', c='r', ls=':')

plt.plot(x, f1_1, label='F1 R1', c='black', ls='-')
plt.plot(x, f1_2, label='F1 R2', c='black', ls=':')

plt.plot(x, bacc_1, label='BACC R1', c='green', ls='-')
plt.plot(x, bacc_2, label='BACC R2', c='green', ls=':')

plt.xlabel('w')
plt.ylabel('performance')
plt.title('Aggregated performance measures as a function of bandwidth')
plt.legend()

# plt.show()
plt.savefig('../paper/figures/agg.eps', format='eps',
            dpi=1000, bbox_inches='tight')

# krnn_r1= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
#                                                             correction= 'r1', w=0.1))

# krnn_none= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
#                                                         correction= None))

krnn_r2 = execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors=1,
                                                            correction='r2', w=0.001))

smote_knn = execute_on_all_datasets(SMOTEWrapper(kNNSimple(5)), 'shufflecv')
acos_knn = execute_on_all_datasets(ACOSWrapper(
    kNNSimple(5), None, 0.001, 10), 'shufflecv')

pnn = execute_on_all_datasets(PNN(1))
krnn1 = execute_on_all_datasets(KRNN(17))
enn = execute_on_all_datasets(ENN(1))
smote_knn = execute_on_all_datasets(SMOTEWrapper(kNNSimple(15)))
bmr_knn = execute_on_all_datasets(
    MetaCostWrapper(KNeighborsClassifier(n_neighbors=5)))
