# flake8: ignore=F401,E501,F404
"""Generate probability of divorce given marriage & individual attributes."""

# TODO: feature_importances
# TODO: http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
# Also look at: KNeighbors Classifier; sklearn.svm.SVC
# (These will be very computationally expensive, though)

import logging
import os
import pickle
import string
import sys

ec2 = sys.platform == 'linux'

if ec2:
    # Deal with barebones EC2 instance
    import matplotlib
    matplotlib.use('Agg')
    del matplotlib
import matplotlib.pyplot as plt


# TODO:
# over_sampling.RandomOverSampler
# vs.
# over_sampling.SMOTE
#

from imblearn import over_sampling
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from src import utils, plots, data

log = logging.getLogger(__name__)

RANDOM_STATE = 444
N_JOBS = -1
TEST_SIZE = 0.33
OVERSAMP_RATIO = 1.25  # TODO
EXT = ''.join(np.random.choice(tuple(string.ascii_letters), size=5))
LEGND_KWARGS = dict(facecolor='wheat', framealpha=0.75, edgecolor='black')

plt.ioff()

ts = '1521559186561633'  # noqa
df = utils.load(ts)

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

df = pd.get_dummies(df, columns=to_dummify, drop_first=True)
y = df.pop('status_fwd')

# Upsample -
# 1. Stratified train-test split
# 2. Upsample X/y train

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Automatic DataFrame -> ndarray conversion
ros = over_sampling.RandomOverSampler(random_state=RANDOM_STATE)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

# NOTE: we don't actually get a timing improvement from fitting on
#       csc sparse array; fits faster on dense arrays in this case.
clf = RandomForestClassifier(n_estimators=2500, n_jobs=N_JOBS,
                             min_samples_split=10, verbose=True,
                             random_state=RANDOM_STATE)
param_grid = {'max_depth': [25, 40],
              'max_features': ['sqrt', 'log2']}
grid = GridSearchCV(clf, param_grid, return_train_score=False,
                    verbose=True, n_jobs=N_JOBS, scoring='roc_auc')
grid.fit(X_resampled, y_resampled)

# Get an idea of our tree depths
mn = np.mean([est.tree_.max_depth for est in grid.best_estimator_.estimators_])
logging.info('Mean tree depth: {:.2f}'.format(mn))
logging.info('Best parameters: {}'.format(grid.best_params_))
logging.info('Best cross-validation score: {:.2f}'.format(grid.best_score_))
logging.info('Test score: {:.2f}'.format(grid.score(X_test, y_test)))
logging.info('Best estimator:\n%s', grid.best_estimator_)

with open(os.path.join(data, 'grid%s_%s.pickle' % (ts, EXT)), 'wb') as f:
    # NOTE: this will be pretty large (2.5 GB+)
    # compress & archive before scp'ing.
    pickle.dump(grid, f, pickle.HIGHEST_PROTOCOL)

y_pred_train = grid.predict(X_resampled)
y_pred = grid.predict(X_test)

cols = ['pred%s' % i for i in (0, 1)]
idx = ['act%s' % i for i in (0, 1)]

_train_cm = confusion_matrix(y_resampled, y_pred_train)
train_cm = pd.DataFrame(_train_cm, columns=cols, index=idx)
norm_train_cm = pd.DataFrame(_train_cm / _train_cm.sum(),
                             columns=cols, index=idx)

_test_cm = confusion_matrix(y_test, y_pred)
test_cm = pd.DataFrame(_test_cm, columns=cols, index=idx)
norm_test_cm = pd.DataFrame(_test_cm / _test_cm.sum(),
                            columns=cols, index=idx)

dc = DummyClassifier(strategy='most_frequent')
dc.fit(X_test, y_test)
dummy_pred = dc.predict(X_test)
_dummy_cm = confusion_matrix(y_test, dummy_pred, labels=[0, 1])
dummy_cm = pd.DataFrame(_dummy_cm, columns=cols, index=idx)
norm_dummy_cm = pd.DataFrame(_dummy_cm / _dummy_cm.sum(),
                             columns=cols, index=idx)

test_fpr = test_cm.iloc[0, 1] / test_cm.iloc[0].sum()
test_tpr = test_cm.iloc[1, 1] / test_cm.iloc[1].sum()

p = grid.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, p)
auc_ = auc(fpr, tpr)

lw = 2
teal = '#%02x%02x%02x' % (128, 191, 183)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.2f)' % auc_)
ax.plot([0, 1], [0, 1], lw=lw, linestyle='--', color=teal)
ax.plot(test_fpr, test_tpr, marker='D', linestyle='', markersize=10,
        color='red')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right", **LEGND_KWARGS)
fig.savefig(os.path.join(plots, '%s_%s.png' % (ts, EXT)))


def custom_predict(y_true, proba_1d, threshold=0.5):
    assert proba_1d.ndim == 1
    if isinstance(threshold, np.ndarray):
        threshold = threshold[:, None]
    return np.where(proba_1d >= threshold, 1., 0.)


pred_cust = custom_predict(y_test, p, threshold=0.25)
_cust_cm = confusion_matrix(y_test, pred_cust)
cust_cm = pd.DataFrame(_cust_cm, columns=cols, index=idx)
norm_cust_cm = pd.DataFrame(_cust_cm / _cust_cm.sum(),
                            columns=cols, index=idx)
cust_fpr = cust_cm.iloc[0, 1] / cust_cm.iloc[0].sum()
cust_tpr = cust_cm.iloc[1, 1] / cust_cm.iloc[1].sum()


# Precision-recall curve;
# One vector of predictions for each threshold.
thresholds = np.arange(0., 0.51, 0.01)
pred_2d = custom_predict(y_test, p, thresholds)


def cust_recall(y_true, pred_2d):
    """Broadcasted recall score with 2d inputs."""
    # tp / (tp + fn)
    y_true = np.asanyarray(y_true)
    tp = ((pred_2d == 1) & (y_true == 1)).sum(axis=1)
    fn = ((pred_2d == 0) & (y_true == 1)).sum(axis=1)
    return tp / (tp + fn)


def cust_precision(y_true, pred_2d):
    """Broadcasted precision score with 2d inputs."""
    # tp / (tp + fp)
    y_true = np.asanyarray(y_true)
    tp = ((pred_2d == 1) & (y_true == 1)).sum(axis=1)
    fp = ((pred_2d == 1) & (y_true == 0)).sum(axis=1)
    return tp / (tp + fp)


rec = cust_recall(y_test, pred_2d)
prec = cust_precision(y_test, pred_2d)
f1 = 2 * (rec * prec) / (rec + prec)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, rec, label='Recall')
ax.plot(thresholds, prec, label='Precision')
ax.plot(thresholds, f1, label='F1 Score')
ax.set_title('Recall, Precision, & F1 as a Function of Threshold')
ax.set_ylabel('Score')
ax.set_xlabel('Decision Threshold')
ax.set_xlim([0.0, 0.50])
ax.set_ylim([0.0, 1.05])
ax.legend(loc=(0.1, 0.60), **LEGND_KWARGS)
plt.savefig(os.path.join(plots, 'pr%s_%s.png' % (ts, EXT)))
