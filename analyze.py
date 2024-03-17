"""
File: analyze.py
----------------
"""
import pandas as pd
from json import load
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

data = []
for file in glob('data/*.json'):
    slug = file.split('/')[-1]
    acl_id = slug[:-5]
    d = load(open(file))
    if not 'funding' in d:
        continue

    funding = d['funding']
    funding = {k: round(v) for k, v in funding.items()}
    d = {**d, **funding, 'acl_id': acl_id}

    del d['funding']
    del d['article']

    data.append(d)

df = pd.DataFrame(data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# -----------------------------------------------------------------------------
# funding by year
# -----------------------------------------------------------------------------
labels = ["defense","corporate","research agency","foundation","none"]
df['decade'] = df.year // 10 % 10 * 10
totals = dict(df.groupby('decade').acl_id.count())

# normalize each year so the row adds up to 1
funding_year = df[df.decade != 80]
funding_year = funding_year.groupby('decade')[labels].sum()
funding_year = funding_year.div(funding_year.sum(axis=1), axis=0)
decade_order = [80, 90, 0, 10, 20]
mapping = {day: i for i, day in enumerate(decade_order)}
key = funding_year.index.map(mapping)
funding_year = funding_year.iloc[key.argsort()]
funding_year.index = funding_year.index.map(lambda x: f'{x:0{2}}s\n({totals[x]} papers)')

# stacked barplot
funding_year.plot(kind='area', stacked=True, rot=0, ax=ax1)

ax1.set_title('Activity of different NLP funders over time')
ax1.set_xlabel('Decade')
ax1.set_ylabel('Relative number of papers funded')

# show legend outside of plot
# ax = plt.gca()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
ax1.legend().remove()

# plt.tight_layout()
# plt.savefig('img/funding_by_decade.png', dpi=300, bbox_inches='tight')
# plt.clf()

# -----------------------------------------------------------------------------
# funding by citation count
# -----------------------------------------------------------------------------
labels = ["defense","corporate","research agency","foundation","none"]
def categorize(citations):
    if 0 <= citations <= 29:
        return '0-29'
    # elif 10 <= citations <= 29:
    #     return '10-29'
    elif 30 <= citations <= 99:
        return '30-99'
    # elif 60 <= citations <= 99:
    #     return '60-99'
    else:
        return '100+'
df['citations'] = df.numcitedby.apply(categorize)
totals = dict(df.groupby('citations').acl_id.count())

# normalize each year so the row adds up to 1
funding_citations = df.groupby('citations')[labels].sum()
funding_citations = funding_citations.div(funding_citations.sum(axis=1), axis=0)

citations_order = ['0-29', '10-29', '30-99', '60-99', '100+']
mapping = {day: i for i, day in enumerate(citations_order)}
key = funding_citations.index.map(mapping)
funding_citations = funding_citations.iloc[key.argsort()]
funding_citations.index = funding_citations.index.map(lambda x: f'{x}\n({totals[x]} papers)')

# stacked barplot
funding_citations.plot(kind='area', stacked=True, rot=0, ax=ax2)

ax2.set_title('Funding of influential NLP research')
ax2.set_xlabel('Citation count')
# ax2.set_ylabel('Funding percentages')

# show legend outside of plot
# ax = plt.gca()
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig('img/funding.png', dpi=300, bbox_inches='tight')
# plt.clf()

# totals = df.groupby('decade')[labels].count()
# funding_citations['total'] = totals
