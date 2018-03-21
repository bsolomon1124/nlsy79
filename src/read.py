import logging
import os
import pickle

import numpy as np
import pandas as pd

from src import downloads, tagsets

log = logging.getLogger(__name__)

# OOP just as a means to get attribute dot-access to intermediate "steps"


class NLSDownload(object):
    """Represents a compartmentalized NLS download and its components.

    Parameters
    ==========
    ts: int or str
        A Unix timestamp such as 1520708978.520341, with or without
        period; as str or int.

    Example
    =======
    >>> from src import read
    >>> ts = '1521387800874926'
    >>> nls = read.NLSDownload(ts)
    """

    def __init__(self, ts):
        self.ts = _check_clean_ts(ts)
        self.metadata = read_metadata(_create_ts_path(ts, 'tagsets', '.csv'))
        assert self.metadata[['myclass', 'year']].duplicated().sum() == 0

        # Doesn't hurt mem usage and we'll eventually need float64
        # Categorical R0000100 will be cast to float in `.melt()`
        self.wide = pd.read_csv(_create_ts_path(ts, 'downloads', '.csv'),
                                dtype=np.float64)
        assert not self.wide['R0000100'].eq(-4).any()
        self.long = self.wide.melt(id_vars='R0000100', var_name='rnum')\
            .set_index('rnum')

        # Unique STATIC elements of `myclass` excluding CASEID
        cl = self.metadata.loc[self.metadata['kind'].eq('STATIC'), 'myclass']
        self.static = np.setdiff1d(cl.values.unique(), ['CASEID'])

        # Merged long-format data with its corresponding metadata
        self.full = self.long.merge(self.metadata, how='left',
                                    left_index=True, right_index=True)

        log.info('NLSDownload[ts=%s]', self.ts)

    def __repr__(self):
        return 'NLSDownload[ts=%s]' % self.ts

    __str__ = __repr__

    def to_pickle(self) -> None:
        path = os.path.join(downloads, self.ts, self.ts + '.pickle')
        with open(path, 'wb') as file:
            try:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
            except OSError:
                # Issue24658
                raise OSError('Failed; obj may be >4GB.')

    @classmethod
    def from_pickle(cls, path_or_ts):
        path = _create_ts_path(path_or_ts, 'downloads', 'pickle')
        with open(path, 'rb') as file:
            return pickle.load(file)

    def get_cdb_info(self, rnum):
        print('\n' + ''.join(_yield_cdb_info(self.ts, rnum)).strip())


# ---------------------------------------------------------------------


def ffill_static(full):
    pt = full[full['kind'].eq('STATIC')].pivot_table(
        index=['year', 'R0000100'], columns='myclass', values='value')
    # NaN is dropped automatially during `.stack()`
    return pt.mask(pt < 0, np.nan)\
        .ffill()\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'value'})


def get_forward_status(frame, n):
    """Returns `1` if R was unmarried for at least 1 month during year."""
    return frame['value'].replace((-4, -5), np.nan)\
        .ffill()\
        .fillna(0.)\
        .resample('A')\
        .prod()\
        .rolling(n)\
        .sum()\
        .shift(-n)\
        .dropna()\
        .lt(n)\
        .astype(np.float64)


def _yield_cdb_info(path_or_ts, rnum):
    """Retrieve codebook info as full string."""
    if '.' not in rnum:
        rnum = rnum[:-2] + '.' + rnum[-2:]
    path = _create_ts_path(path_or_ts, 'downloads', '.cdb')
    with open(path) as file:
        while True:
            line = next(file)
            if line.startswith(rnum):
                yield line
                break
        while True:
            line = next(file)
            if not line.startswith('-' * 10):
                if line.strip():
                    yield line
            else:
                break


def _check_clean_ts(ts):
    return str(ts).replace('.', '')


def _create_ts_path(ts, dirname, ext) -> str:
    """Create absolute path from specified timestamp."""
    ts = _check_clean_ts(ts)
    ext = ext if ext.startswith('.') else '.' + ext
    if dirname == 'downloads':
        # These are nested
        tspath = os.path.join(downloads, ts, ts + ext)
    elif dirname == 'tagsets':
        tspath = os.path.join(tagsets, ts + ext)
    return tspath


def read_metadata(ts):
    subset = ['rnum', 'qname', 'desc', 'year']
    dtypes = dict(
        year=np.float64,
        kind='category',
        myclass='category',
        qname='category',
        desc='category'
        )
    df = pd.read_csv(ts, usecols=range(2, 8), dtype=dtypes)\
        .drop_duplicates(subset=subset)\
        .set_index('rnum')
    return df


def _is_sorted(ser) -> bool:
    # Valid -> [1994, 1994, 1995, np.nan, np.nan]
    # pandas._libs.algos.is_lexsorted and Series.is_monotonic won't work here
    if isinstance(ser, pd.Series):
        ser = ser.values
    return ~(np.diff(ser) < 0).all()
