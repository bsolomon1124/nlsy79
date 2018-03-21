"""File manipulation routine for timestamping & extracting custom tagset."""

import logging
import os
import shutil
import time

from src import tagsets, cstm_tgt

log = logging.getLogger(__name__)


def gen_ts_filename(dirname, ext, timestamp):
    ext = '.' + ext if not ext.startswith('.') else ext
    return os.path.join(
        dirname,
        str(timestamp).replace('.', '') + ext
        )


def copy_and_extract_rnum(verbose=True):
    ts = time.time()
    out_csv = gen_ts_filename(tagsets, ext='.csv', timestamp=ts)
    out_rnum = gen_ts_filename(tagsets, ext='.NLSY79', timestamp=ts)
    if verbose:
        log.info('Copying %s to %s', cstm_tgt, out_csv)
        log.info('Extracting %s to %s', cstm_tgt, out_rnum)

    shutil.copy(cstm_tgt, out_csv, follow_symlinks=False)

    with open(cstm_tgt) as infile, open(out_rnum, 'w') as outfile:
        next(infile)  # no header
        for line in infile:
            outfile.write('%s\n' % line.split(',')[2])
