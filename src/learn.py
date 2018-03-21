"""Generate probability of divorce given marriage & individual attributes."""

# TODO: feature_importances


import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFpr, SelectFromModel
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, auc, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from src import data, plots

log = logging.getLogger(__name__)

RANDOM_STATE = 444
N_JOBS = -1
TEST_SIZE = 0.33

plt.ioff()


def load(ts):
    return pd.read_pickle(os.path.join(data, '%s.pickle' % ts))

ts = '1521559186561633'  # noqa
df = load(ts)


def vc(df, col):
    return df[col].value_counts()


to_dummify = [
    'AFQT-1',
    'CPSIND80',
    'CPSOCC70',
    'DS-11',
    'DS-12',
    'DS-15',
    'ENROLLMENT_STATUS',
    'EXP-1',
    'EXP-7',
    'EXP-8',
    'FAM-15',
    'FAM-21',
    'FAM-26',
    'FAM-27C',
    'FAM-2A',
    'FAM-3',
    'HGC-FATHER',
    'HGC-MOTHER',
    'NET_WORTH',
    'PAY-2A_000002',
    'PAY-2A_000003',
    'POLICE-3E',
    'Q3-2A',
    'REGION',
    'R_REL-1',
    'R_REL-2',
    'R_REL-3',
    'SAMPLE_RACE',
    'SCHOOL-3A_1',
    'TIMEUSECHORES-2B',
    'TIMEUSECHORES-2C',
    'TIMEUSECHORES-2J',
    'TOTAL_INCOME',
    'TNFI_TRUNC',
    'URBAN-RURAL',
    'WKSUEMP-PCY',
    'WKSUEMP-SLI',
    'WORKER_CLASS'
    ]

df_ = df.copy()
df = pd.get_dummies(df, columns=to_dummify, drop_first=True)
y = df.pop('status_fwd')

# Upsample -
# 1. Stratified train-test split
# 2. Upsample X/y train
p = y.value_counts(normalize=True).sort_index().values

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Automatic dataframe -> numpy arr
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)


select = RFE(RandomForestClassifier(n_estimators=500,
                                    random_state=RANDOM_STATE,
                                    n_jobs=N_JOBS, verbose=True),
             n_features_to_select=25, step=6)
select.fit(X_resampled, y_resampled)
# CPU times: user 49min 39s, sys: 29.9 s, total: 50min 9s

mask = select.get_support()
np.savetxt(os.path.join(data, '%s.mask' % ts, mask, delimiter=','))  # -> float

X_resampled_cut = X_resampled[:, mask]
X_test_cut = X_test.values[:, mask]

clf = RandomForestClassifier(n_estimators=1000, n_jobs=N_JOBS,
                             verbose=True)
param_grid = {'max_depth': [10, 25, None],
              'max_features': ['auto', 'sqrt', 'log2']}
p = tuple(itertools.product(*param_grid.values()))
logging.info('Param grid: size = %s', str(len(p)))
logging.info('Total runs')
cv = 4
grid = GridSearchCV(clf, param_grid, cv=cv, return_train_score=False,
                    verbose=True, n_jobs=N_JOBS)  # scoring='recall'
logging.info('Total runs: %s', str(int(len(p) * cv)))

grid.fit(X_resampled_cut, y_resampled)

logging.info('Best parameters: {}'.format(grid.best_params_))
logging.info('Best cross-validation score: {:.2f}'.format(grid.best_score_))
logging.info('Test score: {:.2f}'.format(grid.score(X_test_cut, y_test)))
logging.info('Best estimator:\n%s', grid.best_estimator_)

y_pred_train = grid.predict(X_resampled_cut)
y_pred = grid.predict(X_test_cut)

cols = ['pred%s' % i for i in (0, 1)]
idx = ['act%s' % i for i in (0, 1)]

_train_cm = confusion_matrix(y_resampled, y_pred_train)
train_cm = pd.DataFrame(_train_cm, columns=cols, index=idx)
norm_train_cm = pd.DataFrame(_train_cm / _train_cm.sum(axis=1)[:, None],
                             columns=cols, index=idx)

_test_cm = confusion_matrix(y_test, y_pred)
test_cm = pd.DataFrame(_test_cm, columns=cols, index=idx)
norm_test_cm = pd.DataFrame(_test_cm / _test_cm.sum(axis=1)[:, None],
                            columns=cols, index=idx)

dc = DummyClassifier(strategy='most_frequent')
dc.fit(X_test_cut, y_test)
dummy_pred = dc.predict(X_test_cut)
_dummy_cm = confusion_matrix(y_test, dummy_pred, labels=[0, 1])
dummy_cm = pd.DataFrame(_dummy_cm, columns=cols, index=idx)
norm_dummy_cm = pd.DataFrame(_dummy_cm / _dummy_cm.sum(axis=1)[:, None],
                             columns=cols, index=idx)

test_fpr = test_cm.iloc[0, 1] / test_cm.iloc[0].sum()
test_tpr = test_cm.iloc[1, 1] / test_cm.iloc[1].sum()

p = grid.predict_proba(X_test_cut)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, p)
auc_ = auc(fpr, tpr)

lw = 2
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.2f)' % auc_)
ax.plot([0, 1], [0, 1], lw=lw, linestyle='--')
ax.plot(test_fpr, test_tpr, marker='D', linestyle='', markersize=10,
        color='green')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
fig.savefig(os.path.join(plots, '%s.png' % ts))


def custom_predict(y_true, proba_1d, threshold=0.5):
    assert proba_1d.ndim == 1
    return np.where(proba_1d > threshold, 1., 0.)


# TODO: what does confusion matrix look like at various thresholds?
# i.e.
# pred_cust = custom_predict(y_test, p, threshold=0.2)
# confusion_matrix(y_true, pred_cust)
