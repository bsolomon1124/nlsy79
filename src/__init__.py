"""NLS Investigator data retrieval & manipulation.

Routine
=======

1. Save timestamped copy of `custom_tagset.csv` and extract rnum

>>> from src import extract_tagset
>>> extract_tagset.copy_and_extract_rnum()

2. Use the resulting `<timestamp>.NLSY79` as a tagset on nlsinfo.org/
   NOTE: name download as timestamp i.e. 1520729863887128.
   Source: https://www.nlsinfo.org/investigator/pages/search.jsp?s=NLSY79
   Make sure to drop the six preset-variables.

3. Extract <timestamp>.zip to nlsy79/downloads

4. Instantiate NLSDownload class with new timestamp.

>>> from src import read
>>> ts = '1521387800874926'
>>> nls = read.NLSDownload(ts)
"""

import logging
import os

# This precludes if __name__ == '__main__' scripting...
here = os.path.abspath(os.path.dirname(__file__))

logging.basicConfig(filename=os.path.join(here, '%s.log' % __name__),
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

base = os.path.dirname(here)
tagsets, downloads, data, plots = (os.path.join(base, d) for d in
                                   ('tagsets/', 'downloads/',
                                    'data/', 'plots/'))
cstm_tgt = os.path.join(tagsets, 'custom_tagset.csv')
marstat = os.path.join(base, 'marstat/')
