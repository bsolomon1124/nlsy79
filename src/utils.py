from functools import partial
import logging
import os

import numpy as np
import pandas as pd

from src import downloads, marstat, read, marhist, data
from src.marhist import get_marriage_dates

log = logging.getLogger(__name__)


def load(ts):
    return pd.read_pickle(os.path.join(data, '%s.pickle' % ts))


def cust_cut(s, bins, inval=None, fill=0., exclude=None):
    kwargs = {'labels': np.arange(len(bins) - 1),
              'include_lowest': True,
              'bins': bins
              }
    if not exclude:
        return pd.cut(s.mask(s.isin(inval), fill), **kwargs).astype(np.float64)
    else:
        s_ = s.copy()
        mask = (~s_.isin(exclude)) & (s_.notnull())
        s_.loc[mask] = pd.cut(s_.loc[mask], **kwargs)
        return s_.astype(np.float64)


qcut = partial(pd.qcut, q=4, labels=range(1, 5))
cut = partial(pd.cut, bins=4, labels=range(1, 5))


def cust_qcut(s, exclude=(-1, -2, -3, -4, -5)):
    """Year-relative quantiles."""
    if s.isnull().all():
        return s
    # Not ideal, but fast enough for now...
    s_ = s.copy()
    mask = (~s_.isin(exclude)) & (s_.notnull())
    try:
        s_.loc[mask] = qcut(s_.loc[mask]).astype(np.float64)
    except ValueError:
        try:
            # We have duplicate bin edges.
            # Fall back to uniform bins across range(s_).
            s_.loc[mask] = cut(s_.loc[mask]).astype(np.float64)
        except ValueError:
            # We're screwed
            err = 'Quantiling issue with %s'
            logging.debug(err, s_.name)
            raise ValueError(err % s_.name)
    return s_


def custom_replace(s, to_replace):
    """Replace with specified scalars or mapped aggregates."""
    ge = s[s.ge(0)]
    if ge.isnull().all():
        # We have no valid values to fill with.
        # Applies with STATIC variables.
        new = {k: (np.nan if isinstance(v, str) else v)
               for k, v in to_replace.items()}
    else:
        new = {}
        for k, v in to_replace.items():
            if v == 'mean':
                new[k] = ge.mean()
            elif v == 'med':
                new[k] = ge.median()
            elif v == 'mc':
                # Most-common
                new[k] = ge.value_counts().idxmax()
            else:
                new[k] = v
    return s.replace(new)


def test_age_1m(ts, factor=0.10):
    """Confirm age-at-first-marriage checks out."""
    dob = pd.read_csv(os.path.join(downloads, 'DOB/DOB.csv'),
                      dtype={'R0000300': 'object', 'R0000500': 'object'})
    dob = pd.Series(pd.to_datetime(dict(year='19' + dob['R0000500'],
                                        month=dob['R0000300'], day=1)).values,
                    index=dob['R0000100'])
    nls = read.NLSDownload(ts)
    mh = marhist.MarHist()
    assert mh.merged.duplicated(['year', 'R0000100']).sum() == 0
    age = nls.full.pivot_table(columns='myclass', values='value',
                               index=['R0000100', 'year']).loc[:, 'AGE1M']\
        .dropna()
    age.index = age.index.get_level_values(0)
    m = get_marriage_dates(os.path.join(marstat, 'marriage.csv'))['marriage1']
    diff = age / ((m - dob).dt.days / 365)
    assert diff[np.abs(1 - diff) > factor].empty


def all_invalid(ser, filt=False):
    """Count, by year, of >= 0 values."""
    ct = ser.groupby(level=1).apply(lambda x: x.ge(0).sum())
    if filt:
        return ct[ct.eq(0)]
    return ct


# def rowise_quantile(ser, to_replace=(-5, -3), value=np.nan, **kwargs):
#     ppt = ser.unstack(level=0)
#     if to_replace:
#         ppt = ppt.replace(to_replace, value=value)
#     ppt = ppt.apply(_cust_qcut, axis=1, **kwargs)
#     return ppt.stack().swaplevel(0, 1).reindex(ser.index)


# Codes; see also: http://bit.ly/2Iwe13M
# ---------------------------------------------------------------------
# Industry:
#  10 TO 31: 010-031 AGRICULTURE,FORESTRY AND FISHERIES
#  40 TO 50: 040-050 MINING
#  60: 060     CONSTRUCTION
# 100 TO 392: 100-392 MANUFACTURING
# 400 TO 472: 400-472 TRANSPORTATION,COMMUNICATION,PUBLIC UTILITIES
# 500 TO 571: 500-571 WHOLESALE TRADE
# 580 TO 691: 580-691 RETAIL TRADE
# 700 TO 712: 700-712 FINANCE, INSURANCE, AND REAL ESTATE
# 721 TO 760: 721-760 BUSINESS AND REPAIR SERVICES
# 761 TO 791: 761-791 PERSONAL SERVICES
# 800 TO 802: 800-802 ENTERTAINMENT AND RECREATION SERVICES
# 812 TO 892: 812-892 PROFESSIONAL AND RELATED SERVICES
# 900 TO 932: 900-932 PUBLIC ADMINISTRATION
#
# Occupational:
#   1 TO 195: 001-195 PROFESSIONAL,TECHNICAL AND KINDRED
# 201 TO 245: 201-245 MANAGERS,OFFICIALS AND PROPRIETORS
# 260 TO 285: 260-285 SALES WORKERS
# 301 TO 395: 301-395 CLERICAL AND KINDRED
# 401 TO 575: 401-575 CRAFTSMEN,FOREMEN AND KINDRED
# 580 TO 590: 580-590 ARMED FORCES
# 601 TO 715: 601-715 OPERATIVES AND KINDRED
# 740 TO 785: 740-785 LABORERS, EXCEPT FARM
# 801 TO 802: 801-802 FARMERS AND FARM MANAGERS
# 821 TO 824: 821-824 FARM LABORERS AND FOREMAN
# 901 TO 965: 901-965 SERVICE WORKERS, EXCEPT PRIVATE HOUSEHOLD
# 980 TO 984: 980-984 PRIVATE HOUSEHOLD
