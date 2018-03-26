"""Marital history time-series munging.

>>> from src import marhist
>>> mh = marhist.MarHist()
>>> mh.to_pickle()
"""

import functools
from itertools import chain
import os
import pickle

import numpy as np
import pandas as pd
from pandas.tseries import offsets

from src import marstat


class MarHist(object):
    def __init__(self, n=5):
        self.n = n
        self.marhist = get_marriage_history()
        self.bin_status = fwd_binary_status(self.marhist, n=self.n)
        self.yearend_status = yearend_status(self.marhist)
        self.merged = merge(self.bin_status, self.yearend_status)

    def to_pickle(self, name=None):
        if not name:
            name = 'n%s' % self.n
        path = os.path.join(marstat, name + '.pickle')
        with open(path, 'wb') as file:
            try:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
            except OSError:
                raise OSError('Failed; obj > 4GB.')

    @classmethod
    def from_pickle(self, path=None, n=None):
        if not path:
            path = os.path.join(marstat, 'n%s' % n + '.pickle')
        with open(path, 'rb') as file:
            return pickle.load(file)


def read_marriage_dates(path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col='R0000100', dtype=np.float64)
    mask = frame.le(0)
    return frame.mask(mask, np.nan).dropna(how='all')


def concat_yymm(mm: np.ndarray, yy: np.ndarray, day=1):
    # Will yield NaT if either `yy` or `mm` are NaN
    return pd.to_datetime({'month': mm, 'year': yy, 'day': day})


def get_marriage_dates(frame, topic='marriage') -> pd.DataFrame:
    if isinstance(frame, str):
        frame = read_marriage_dates(frame)
    pairs = frame.columns.values.reshape(-1, 2)
    return pd.DataFrame({'%s%s' % (topic, i): concat_yymm(frame[mm], frame[yy])
                         for i, (mm, yy) in enumerate(pairs, 1)})


def get_marriage_mask(dates: pd.DataFrame, min_start='1970-01-01',
                      max_end='2017-01-01') -> np.ndarray:
    # Result is <inclusive, exclusive>
    dates = dates.copy()
    min_start = pd.to_datetime(min_start)
    max_end = pd.to_datetime(max_end)

    # A hack to validate NaT comparisons
    dates = dates.fillna(max_end + offsets.MonthBegin())
    idx = pd.date_range(min_start, max_end, freq='MS')[:, None]
    marriage = dates['start'].values
    divorce = dates['end'].values
    mask = (idx >= marriage) & (idx < divorce)
    return pd.DataFrame(mask, index=idx.squeeze(), columns=dates.index)


@functools.lru_cache(maxsize=32)
def get_marriage_history() -> pd.DataFrame:
    m = get_marriage_dates(os.path.join(marstat, 'marriage.csv'))
    d = get_marriage_dates(os.path.join(marstat, 'divorce.csv'),
                           topic='divorce').reindex(m.index)

    # We have a couple hundred who listed no end to marriage 't' but
    #     a beginning to marriage 't + 1', through invalid skip (-3)
    inval1 = ((m.marriage2.notnull()) & (d.divorce1.isnull()))
    inval2 = ((m.marriage3.notnull()) & (d.divorce2.isnull()))
    inval = inval1 | inval2
    m, d = m.loc[~inval], d.loc[~inval]
    m.drop('marriage3', axis=1, inplace=True)
    r1, r2 = (pd.concat((i, j), axis=1) for (_, i), (__, j) in
              zip(m.iteritems(), d.iteritems()))
    r1.columns, r2.columns = [['start', 'end']] * 2
    masks = (get_marriage_mask(r) for r in (r1, r2))
    return np.logical_or(*masks).astype(np.float64)


# Can't cache with a DataFrame input (not hashable)
def fwd_binary_status(frame, n: int) -> pd.DataFrame:
    """Will R be unmarried for at least 1 month for n-year period starting
    at Jan 1 the next year?  1 (positive) if so.

    n: *years* forward
    """

    months = 12 * n
    fwd = frame\
        .rolling(months)\
        .sum()\
        .shift(-months)\
        .dropna()\
        .lt(months)\
        .astype(np.float64)
    fwd = fwd[fwd.index.month == 1].stack().reset_index().rename(
        columns={'level_0': 'year', 0: 'status'})
    fwd['year'] = fwd['year'].dt.year - 1
    return fwd


def yearend_status(frame) -> pd.DataFrame:
    res = get_marriage_history()
    res = res[res.index.month == 12]
    res.index = range(res.index.year.min(), res.index.year.max() + 1)
    return res.stack().reset_index().rename(columns={'level_0': 'year',
                                                     0: 'status'})


def merge(bin_status, y_end_status) -> pd.DataFrame:
    merged = bin_status.merge(y_end_status, on=['year', 'R0000100'],
                              suffixes=('_fwd', ''))
    # Drop biennial & early yrs now just for sake of space
    excl = chain(range(1995, 2017, 2), range(1960,  1978))
    return merged[(merged.status==1) & (~merged.year.isin(excl))]\
        .drop('status', axis=1)
