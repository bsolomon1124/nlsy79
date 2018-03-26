# flake8: ignore=E501

import logging
import os

import numpy as np
import pandas as pd

from src import data, marhist, marstat, read
from src.mapping import mapper, religions
from src.utils import custom_replace, cust_qcut, cust_cut
from src.marhist import get_marriage_dates

log = logging.getLogger(__name__)


def main(ts):
    nls = read.NLSDownload(ts)

    # `pt` has a lexsorted MultiIndex of (R0000100, year) with
    #     other rnums as fields.
    pt = nls.full.pivot_table(columns='myclass', values='value',
                              index=['R0000100', 'year'])
    logging.info(pt.__str__)
    logging.info('Pivot table created ...')
    assert pt.index.is_unique
    assert pt.index.is_lexsorted

    # A few simplifications.  -1 and -2 are treated identically and
    #     there isn't any validity to a -5 (non-interview).
    pt.replace(to_replace=(-5, -1), value=(np.nan, -2.), inplace=True)

    # We can't transform entire DataFrame.  Kludgy, but need to make tailored
    #     replacements on a per-column, year-relative basis.  Otherwise
    # we have major look-ahead bias on statistic calculation.
    logging.info('Mapping to custom replacements ...')

    grouped = pt.groupby(level=1, sort=False)
    assert not set(mapper.keys()).difference(pt.columns)
    for col, v in mapper.items():
        pt.loc[:, col] = grouped[col].transform(custom_replace, to_replace=v)

    # Now we can get year-relative quartiles, again on a per-variable basis
    #     because qcut applies to Series
    to_bin = (
        'NET_WORTH',
        'TNFI_TRUNC',
        'TOTAL_INCOME',
        'AFQT-1',
        'DS-11',
        'DS-12',
        'DS-15'
        )  # TODO: there should be others...
    grouped = pt.groupby(level=1, sort=False)
    for col in to_bin:
        pt.loc[:, col] = grouped[col].transform(cust_qcut)

    # Custom binning (code-based, not quantiling) on industry/occupation.
    # See also: http://bit.ly/2Iwe13M.
    logging.info('Custom binning ...')
    bins = {
        'CPSIND80': {
            'bins': [0, 9, 31, 60, 77, 398, 479, 699, 761, 799, 810, 899,
                     np.inf],
            'inval': (-1, -2, -4),
            'fill': 0.
            },
        'CPSOCC70': {
            'bins': [-0.1, 1, 195, 245, 285, 399, 599, 715, 785, 815, 899,
                     970],
            'inval': (-1, -2, -4),
            'fill': 0.
            },
        # TODO: sure we're okay here??
        'WKSUEMP-PCY': {
            'bins': [-0.1, 0.1, 13, 26, np.inf],
            'inval': (-1, -2, -4),
            'fill': np.nan
            },
        'WKSUEMP-SLI': {
            'bins': [-0.1, 0.1, 13, 26, np.inf],
            'inval': (-1, -2, -4),
            'fill': np.nan
            },
        'HGC': {
            'bins': [0, 6, 11.5, 12.5, 16, np.inf],
            'exclude': (-4,)
            },
        'HGC-FATHER': {
            'bins': [0, 6, 11.5, 12.5, 16, np.inf],
            'exclude': (-4,)
            },
        'HGC-MOTHER': {
            'bins': [0, 6, 11.5, 12.5, 16, np.inf],
            'exclude': (-4,)
            },
        # *Ever* stopped
        'POLICE-1A': {
            'bins': [-0.1, 0.1, 2, 5, 9, np.inf],
            'exclude': (-4,)
            },
        # Stopped over last calendar year
        'POLICE-1B': {
            'bins': [-0.1, 0.1, 2, 9, np.inf],
            'exclude': (-4,)
            },
        'TIMEUSETV-2_HRS': {
            'bins': [-0.1, 0.1, 8, 25, 40, np.inf],
            'exclude': (-4,)
            }
        }

    # Here, we don't need to be year-relative because bins are static.
    for col, d in bins.items():
        pt.loc[:, col] = cust_cut(pt[col], **d)

    # We don't care that you have 8 or 9 or 10 kids,
    #    just that you have way too many kids.
    cols = ['FER-3', 'FER-1A', 'FER-1B', 'NUMKID', 'NUMCH', 'FAMSIZE']
    pt.loc[:, cols] = pt.loc[:, cols].clip_upper(threshold=7.)

    # Another case where survey designer made the wonderful choice to
    #     change the response encodings over time... (Codes are reversed)
    roles = ('WOMENS-ROLES_00000%s' % i for i in range(1, 9))
    for n in roles:
        # Note this will mask anything we don't specify to NaN (-4)
        pt.loc[(slice(None), 2004), n] = pt.loc[(slice(None), 2004), n].map(
            {1: 4., 2: 3., 3: 2., 1: 4., 8: 2.})

    logging.info('Re-encoding religion ...')
    # Simplify religion encoding a bit.  We have changing religion codes
    #    over time.  (No joke.)  See http://bit.ly/2plMh9q.
    pt.loc[(slice(None), 1979), 'R_REL-1'] = \
        pt.loc[(slice(None), 1979), 'R_REL-1'].map(religions[1979])

    for year in (1979, 1982, 2000, 2012):
        # Anything not included in mapper keys goes to NaN
        pt.loc[(slice(None), year), 'R_REL-2'] = \
            pt.loc[(slice(None), year), 'R_REL-2'].map(religions[year])

    pt['R_REL-3'] = pt['R_REL-3'].map({1: 1, 2: 2, 4: 4, 6: 6, 5: 4., 3: 2})

    # A good time to check on where we still have invalid values
    #     and why we have them.
    filt = pt.isin((-1, -2, -3, -4, -5))
    inval = filt.loc[:, filt.sum(axis=0).gt(0)].columns

    # Places where -4 encodes something meaningful
    # Or where we haven't binarized yet
    retained_neg = (
        'SCHOOL-3A_1',
        'SEPARATED',    # binary
        'AFQT-1',
        'EXP-1',
        'EXP-7',
        'EXP-8',
        'FAM-27C',
        'FAM-26',
        'FFER-142',  # binary (sex ed)
        'MFER-31',  # binary (sex ed)
        'FFER-92',  # digitized
        'MFER-15',  # digitized
        'NET_WORTH',
        'TIMEUSECHORES-2B',
        'TIMEUSECHORES-2C',
        'TIMEUSECHORES-2J',
        'H40-BPAR-1',
        'H40-BPAR-6',
        'HGC-FATHER',
        'HGC-MOTHER',
        'PAY-2A_000002',
        'PAY-2A_000003',
        'POLICE-3E',
        'Q3-2A',
        'REGION',
        'WORKER_CLASS',
        'TOTAL_INCOME',
        'TNFI_TRUNC',
        'URBAN-RURAL'
        )

    assert inval.difference(retained_neg).empty

    logging.info('Filling forward ...')
    pt = pt.groupby(level=0, sort=False).ffill()

    # Binary field derivation
    logging.info('Deriving binary fields ...')
    pt.columns = pt.columns.add_categories('ANY_SEX_ED')
    pt.loc[:, 'ANY_SEX_ED'] = ((pt['FFER-142'] == 1)
                               |
                               (pt['MFER-31'] == 1)).astype(np.float64)

    binmap = (
        ('ANY_AFDC', 'AFDC_TOTAL'),
        ('ANY_FDSTMPS', 'FDSTMPS_TOTAL'),
        ('ANY_KID', 'NUMKID'),
        ('ANY_WELFARE', 'WELFARE_TOTAL'),
        )

    # Don't use np.nonzero or truthiness here.  Will catch NaN as True.
    #     This is fast as-is.
    newnames, cols = map(list, zip(*binmap))
    pt.columns = pt.columns.add_categories(newnames)
    loc = pt.loc[:, cols].values
    mask = (loc == 0) | np.isnan(loc) | np.isin(loc, np.arange(-1, -6, -1))
    binarr = np.where(mask, 0., 1.)

    # Can't assign multiple columns with loc
    pt = pd.concat((pt, pd.DataFrame(binarr, columns=newnames,
                                     index=pt.index)), axis=1)

    pt.loc[:, 'FL1M1B'] = pt['FL1M1B'].eq(0).astype(np.float64)
    pt.loc[:, 'SEPARATED'] = pt['SEPARATED'].eq(2).astype(np.float64)

    pt.columns = pt.columns.add_categories(['IS_VIRGIN', 'YOUNG_SEX',
                                            'LATE_SEX'])
    young = np.arange(0, 17, dtype=np.float64)
    sex = pd.concat((pt['FFER-92'], pt['MFER-15']), axis=1)
    pt.loc[:, 'IS_VIRGIN'] = sex.eq(-4).all(axis=1).astype(np.float64)
    pt.loc[:, 'YOUNG_SEX'] = sex.isin(young).any(axis=1).astype(np.float64)
    pt.loc[:, 'LATE_SEX'] = sex.ge(20).any(axis=1).astype(np.float64)

    for r in ('ROTTER-1A', 'ROTTER-3A', 'SAMPLE_SEX'):
        pt.loc[:, r] = pt[r].where(pt[r].eq(1.), 0.)

    # In cases where we've filled forward (-3) response, correct for
    #     0-1-0 sequences.  I.e. [0, 0, 1, 0, 0, 1] -> [0, 0, 1, 1, 1, 1]
    # Want to do the opposite for other fields (H40-BPAR-1) which are inverted
    logging.info('Correcting cumulative binary responses ...')
    pt.loc[:, 'H40-BPAR-1'] = 1. - pt['H40-BPAR-1']
    pt.loc[:, 'H40-BPAR-6'] = 1. - pt['H40-BPAR-6']
    grouped = pt.groupby(level=0, sort=False)
    for col in ('ANYABORT', 'H40-BPAR-1', 'H40-BPAR-6'):
        pt.loc[:, col] = grouped[col].transform(
            lambda s: s.cumsum().gt(0).astype(np.float64))

    # First-differencing
    to_diff = (
        'Q11-9',
        'NUMKID',
        'JOBSNUM',
        'TNFI_TRUNC'
        )
    pt.columns = pt.columns.add_categories(['DD_%s' % col for col in to_diff])
    grouped = pt.groupby(level=0, sort=False)
    for col in to_diff:
        pt.loc[:, 'DD_%s' % col] = grouped[col].pct_change()

    # Merge on (R0000100, year)
    # ---------------------------------------------------------------------

    mh = marhist.MarHist()
    assert mh.merged.duplicated(['year', 'R0000100']).sum() == 0

    pt.columns = pt.columns.add_categories('fwd_status')
    full = mh.merged.set_index(['R0000100', 'year'])\
        .merge(pt, how='left', left_index=True, right_index=True)\
        .dropna()

    # We don't need to worry about CASEID/R0000100; it's in the Index
    drop = [
        'H40-BPAR-1',
        'H40-BPAR-6',
        'FFER-92',
        'MFER-15',
        'FFER-142',
        'MFER-31'
        ]

    # A few last derivations
    m = get_marriage_dates(os.path.join(marstat, 'marriage.csv'))
    marnum = full[['AGEATINT']].reset_index().merge(m.reset_index(),
                                                    on='R0000100',
                                                    how='left')
    is_second_marriage = (marnum.marriage2.dt.year < marnum.year).values
    full['MARLENGTH'] = np.where(is_second_marriage,
                                 marnum.year - marnum.marriage2.dt.year,
                                 marnum.year - marnum.marriage1.dt.year)
    full['SECOND_MARRIAGE'] = is_second_marriage
    full['AGEAT_CUR_MARRIAGE'] = full['AGEATINT'] - full['MARLENGTH']
    # Missing a handful of marriage1 dates
    full.dropna(inplace=True)

    full.drop(drop, axis=1, inplace=True)
    dtypes = full.get_dtype_counts().to_string().replace('\n', '\t')
    path = os.path.join(data, '%s.pickle' % ts)
    log.info('Pickling to %s', path)
    log.info('Dtypes: %s', dtypes)
    full.to_pickle(path)
