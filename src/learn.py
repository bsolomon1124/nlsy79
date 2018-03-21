# flake8: ignore=F401,E501
"""Generate probability of divorce given marriage & individual attributes."""

# TODO: feature_importances

import logging
import os
import pickle

import matplotlib.pyplot as plt  # careful with backend in EC2
import numpy as np
import pandas as pd

from imblearn import over_sampling

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
OVERSAMP_RATIO = 1.25  # TODO

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

# Automatic DataFrame -> ndarray conversion
# TODO: we can pass dict to `ratio=` i.e...
# vc = y.value_counts()[0]
#ratio = {0: vc, 1: int(OVERSAMP_RATIO * vc)}
ros = over_sampling.RandomOverSampler(random_state=RANDOM_STATE)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)


# with open(os.path.join(data, 'grid%s.pickle' % ts), 'rb') as f:
#     grid = pickle.load(f)

# TODO: add min_samples_split to grid search
# Account for the fact that oversampling causes duplicate records
clf = RandomForestClassifier(n_estimators=2500, n_jobs=N_JOBS,
                             min_samples_split=10, verbose=True)
param_grid = {'max_depth': [10, 25, None],
              'max_features': ['auto', 'sqrt', 'log2']}
grid = GridSearchCV(clf, param_grid, cv=4, return_train_score=False,
                    verbose=True, n_jobs=N_JOBS)
grid.fit(X_resampled, y_resampled)

logging.info('Best parameters: {}'.format(grid.best_params_))
logging.info('Best cross-validation score: {:.2f}'.format(grid.best_score_))
logging.info('Test score: {:.2f}'.format(grid.score(X_test, y_test)))
logging.info('Best estimator:\n%s', grid.best_estimator_)

with open(os.path.join(data, 'grid%s.pickle' % ts), 'wb') as f:
    # NOTE: this will be pretty large (2.5 GB+)
    # compress & archive before scp'ing.
    pickle.dump(grid, f, pickle.HIGHEST_PROTOCOL)

y_pred_train = grid.predict(X_resampled)
y_pred = grid.predict(X_test)

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
dc.fit(X_test, y_test)
dummy_pred = dc.predict(X_test)
_dummy_cm = confusion_matrix(y_test, dummy_pred, labels=[0, 1])
dummy_cm = pd.DataFrame(_dummy_cm, columns=cols, index=idx)
norm_dummy_cm = pd.DataFrame(_dummy_cm / _dummy_cm.sum(axis=1)[:, None],
                             columns=cols, index=idx)

test_fpr = test_cm.iloc[0, 1] / test_cm.iloc[0].sum()
test_tpr = test_cm.iloc[1, 1] / test_cm.iloc[1].sum()

p = grid.predict_proba(X_test)[:, 1]
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
