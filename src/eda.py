#
# TODO:
# - Aggregation of cumulative divorce rate
#

import os

import matplotlib.pyplot as plt

from src import plots, utils

plt.ioff()


ts = '1521559186561633'  # noqa
df = utils.load(ts)


# Distribution of age at first marriage
# ---------------------------------------------------------------------

mask = ~df.index.get_level_values(0).duplicated(keep='first')
age1m = df.loc[mask, 'AGE1M']
assert age1m.index.get_level_values(0).is_unique
med, mean = age1m.agg(['median', 'mean'])
text = '$\mu=%.1f$\n$\mathrm{median}=%.1f$' % (mean, med)

plt.close('all')
ax = age1m.hist(bins=age1m.nunique())
ax.set_ylabel('Number of Respondents')
ax.set_xlabel('Age at First Marriage')
ax.set_title('Distribution of Ages at First Marriage')
ax.text(0.7, 0.75, text, transform=ax.transAxes, fontsize=14, bbox=dict(
    boxstyle='round', facecolor='wheat', alpha=0.75, ec='black'))
plt.savefig(os.path.join(plots, '%s_hist.png' % ts))


# Forward-looking statistics
# ---------------------------------------------------------------------

y = df.pop('status_fwd')
grouped = y.groupby('year')
prop = grouped.sum().div(grouped.count())
