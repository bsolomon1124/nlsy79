# flake8: noqa

from numpy import nan


mapper = {
    'AFDC_TOTAL': {-2: 'mean', -3: 'mean', -4: 0.},
    # Keep -4
    'AFQT-1': {-3: 'mean'},
    # We have some -3s that we can't derive here (260 to be exact)
    # TODO: may want to force-3/2/1 altogether to NaN
    'AGE1M': {-999: nan, -3: 'med', -2: 'med'},
    'AGEATINT': {-3: nan, -4: nan},
    'ANYABORT': {-4: 0., -2: 1., -3: 'mc'},
    # 'CASEID':
    'CPSIND80': {-2: 850.},
    'CPSOCC70': {-2: 850.},
    # 'CURR_PREG': {-4: 0., -2: 1., -3: 'mc'},  # deleted - fem. only
    # 'DELIN-1': {-2: 2., -3: 'mc'},  # deleted - R >= 17
    'DELIN-5': {-2: 2., -3: 'mc'},
    'DS-11': {-4: 0., -2: 2., -3: 'med'},
    'DS-12': {-4: 0., -2: 2., -3: 'med'},
    'DS-15': {-4: 0., -2: 2., -3: 'med'},
    'ENROLLMENT_STATUS': {i: 'mc' for i in (-4, -3, -2)},
    # Keep -2 ("don't know" is meaningful here)
    'EXP-1': {-3: 'mc'},
    'EXP-7': {-3: 'mc', -4: 1.},
    'EXP-8': {-3: 'mc', -2: 3.},
    'FAM-15': {-2: 3., -3: 'mc'},
    'FAM-21': {-2: 3., -3: 'mc'},
    # Keep -4 ("R with version C of HH record card")
    'FAM-26': {-3: 'mc'},  # keep -2 here also
    'FAM-27C': {-2: 0., -3: 'mc'},
    'FAM-2A': {-3: 'mc'},
    'FAM-3': {-3: 'mc'},
    'FAMSIZE': {-3: nan},
    'FDSTMPS_TOTAL': {-2: 500., -3: 'mean', -4: 0.},
    'FER-1A': {-2: 0., -3: 'mc'},
    'FER-1B': {-2: 0., -3: 'mc'},
    'FER-3': {-2: 0., -3: 'mc'},
    'MFER-31': {-4: 0., -2: 0},
    'FFER-142': {-4: 0., -2: 0},
    # Keep -4 ("R has had sexual intercourse")
    'FFER-92': {-2: 14., -3: 'mc'},
    'MFER-15': {-2: 14., -3: 'mc'},
    # 'FL1M1B':
    # Keep -4 ("R without father/father figure in the HH")
    'H40-BPAR-1': {-2: 0., -3: 'mc'},
    'H40-BPAR-6': {-2: 0., -3: 'mc'},
    'HGC': {-3: nan},
    # Careful - *a lot * of -1/-2/-3s here
    # Keep -4 ("R with father/father figure")
    'HGC-FATHER': {-3: nan, -2: 9.},
    'HGC-MOTHER': {-3: nan, -2: 9.},
    'JOBSNUM': {-3: nan, -2: nan},
    'MILWK-PCY': {-3: nan, -4: 0.},
    'NET_WORTH': {-3: 'mean'},
    'NUMCH': {-2: nan, -3: nan},
    'NUMKID': {-2: nan, -3: nan, -4: 0.},
    # Keep -4s ("R who would not accept job offer at [lower increment]")
    'PAY-2A_000001': {-3: 0., -2: 0.},
    'PAY-2A_000002': {-3: 0., -2: 0.},
    'PAY-2A_000003': {-3: 0., -2: 0.},
    'POLICE-1A': {-4: 0., -3: 2.},
    'POLICE-1B': {-4: 0., -3: 2.},
    'POLICE-3': {-3: 'mc', -2: 1.},
    'POLICE-3D_000001': {-2: 0., -4: 0., -3: 0.},
    # Keep -4 ("R has been convicted of crime other
    #           than minor traffic offense")
    'POLICE-3E': {-3: 0.},
    'POVSTATUS': {-3: nan, },
    'Q11-9': {-2: nan, -3: nan, -4: nan},
    'Q12-3': {-2: 1., -3: 'mc', -4: 0.},
    # Keep -4 ("R attended/enrolled in school prior to DLI")
    'Q3-2A': {-2: 13., -3: nan, 11: 13., 12: 13., 7: 8., 5: 4.},
    'R_REL-1': {-2: 0., -3: 'mc', -4: nan},
    'R_REL-2': {-2: 0., -3: 'mc', -4: nan},
    'R_REL-3': {-2: 0., -3: 1., -4: nan},
    # Keep -4s ("no records")
    'REGION': {-2: 'mc', -3: nan},
    'ROSENBERG_ESTEEM_000001': {-2: 3., -3: 'mc'},
    'ROSENBERG_ESTEEM_000002': {-2: 3., -3: 'mc'},
    'ROSENBERG_ESTEEM_000003': {-2: 2., -3: 'mc'},
    'ROSENBERG_ESTEEM_000004': {-2: 3., -3: 'mc'},
    'ROSENBERG_ESTEEM_000005': {-2: 2., -3: 'mc'},
    'ROSENBERG_ESTEEM_000006': {-2: 3., -3: 'mc'},
    'ROSENBERG_ESTEEM_000007': {-2: 2., -3: 'mc'},
    'ROSENBERG_ESTEEM_000008': {-2: 2., -3: 'mc'},
    'ROSENBERG_ESTEEM_000009': {-2: 2., -3: 'mc'},
    'ROSENBERG_ESTEEM_000010': {-2: 2., -3: 'mc'},
    'ROTTER-1A': {-2: 'mc', -3: 'mc'},
    'ROTTER-3A': {-2: 'mc', -3: 'mc'},
    # 'SAMPLE_RACE':
    # 'SAMPLE_SEX':
    'SCHOOL-3A_1': {-2: 3., -3: 2.},
    # 'SEPARATED':
    # Keep -4s ("R does not live in group quarters")
    'TIMEUSECHORES-2B': {-2: 1, -3: 'mc', 3: 2, 4: 5},
    'TIMEUSECHORES-2C': {-2: 1, -3: 'mc', 3: 2, 4: 5},
    'TIMEUSECHORES-2J': {-2: 1, -3: 'mc', 3: 2, 4: 5},
    'TIMEUSETV-2_HRS': {-2: 'mean', -3: 'mean'},
    # TODO: choice of whether -2 -> nan, mean, or remains
    'TNFI_TRUNC':  {-3: 'mean'},
    'TOTAL_INCOME':  {-3: 'mean'},
    'URBAN-RURAL': {-3: nan},
    # A case where variable can be kept continuous
    'WELFARE_TOTAL': {-2: 'mean', -3: 'mean', -4: 0.},
    'WKSUEMP-PCY': {-2: 'mean', -3: nan},
    'WKSUEMP-SLI': {-2: 'mean', -3: nan},
    # Encoding of responses changes over time ...
    # Don't convert 8 here (*yet*) because of inversion on final year
    'WOMENS-ROLES_000001': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000002': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000003': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000004': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000005': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000006': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000007': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WOMENS-ROLES_000008': {-2: 2., -3: 'mc', -4: nan, 9: 'mc'},
    'WORKER_CLASS': {-2: 1., -3: 'mc'}
    }

# 0   None
# 1   Protestant
# 2   Baptist
# 3   Episcopalian
# 4   Lutheran
# 5   Methodist
# 6   Presbyterian
# 7   Roman Catholic
# 8   Jewish
# 98  Oriental/East Europe (Baha'i, Buddhist, Russian Orthodox...)
# 99  Reformed/born-again/evangelical

base = dict(enumerate(range(9)))
oriental = {i: 98 for i in range(100, 200)}
reform = {i: 99 for i in range(200, 400)}

religions = {
    # http://bit.ly/2plMh9q
    1979: {**base, **oriental, **reform},
    1982: {
        1: 1,
        2: 7,
        3: 8,
        4: 0,
        5: 99
        },
    2000: {
        1: 1,
        2: 2,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 99,
        10: 0,
        },
    2012: {
        1: 1,
        2: 2,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 99,
        10: 0,
        12: 1,
        11: 7
        }
    }
