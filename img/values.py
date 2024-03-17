import pandas as pd
from json import load
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

data = []
for file in glob("data/*.json"):
    slug = file.split("/")[-1]
    acl_id = slug[:-5]
    d = load(open(file))
    if "values" not in d or not d["values"]:
        continue

    values = d["values"]
    espoused_values = list(values.keys())
    espoused_values = {k: 1 for k in espoused_values}

    d = {
        "acl_id": acl_id,
        **espoused_values,
        "year": d["year"],
        "numcitedby": d["numcitedby"],
    }

    data.append(d)

df = pd.DataFrame(data).fillna(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# -----------------------------------------------------------------------------
# funding by year
# -----------------------------------------------------------------------------
labels = [
    "building on past work",
    "novelty",
    "performance",
    "reproducibility",
    "ease of implementation",
    "fairness",
]
df["decade"] = df.year // 10 % 10 * 10
totals = dict(df.groupby("decade").acl_id.count())

# normalize each year so the row adds up to 1
values_year = df[df.decade != 80]
values_year = values_year.groupby("decade")[labels].sum()
values_year = values_year.div(values_year.sum(axis=1), axis=0)
decade_order = [80, 90, 0, 10, 20]
mapping = {day: i for i, day in enumerate(decade_order)}
key = values_year.index.map(mapping)
values_year = values_year.iloc[key.argsort()]
values_year.index = values_year.index.map(lambda x: f"{x:0{2}}s\n({totals[x]} papers)")

# stacked barplot
values_year.plot(kind="area", stacked=True, rot=0, ax=ax1)

ax1.set_title("Values espoused by papers over time")
ax1.set_xlabel("Decade")
ax1.set_ylabel("Relative number of papers")

# show legend outside of plot
# ax = plt.gca()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
ax1.legend().remove()


# -----------------------------------------------------------------------------
# funding by citation count
# -----------------------------------------------------------------------------
def categorize(citations):
    if 0 <= citations <= 29:
        return "0-29"
    # elif 10 <= citations <= 29:
    #     return '10-29'
    elif 30 <= citations <= 99:
        return "30-99"
    # elif 60 <= citations <= 99:
    #     return '60-99'
    else:
        return "100+"


df["citations"] = df.numcitedby.apply(categorize)
totals = dict(df.groupby("citations").acl_id.count())

# normalize each year so the row adds up to 1
value_citations = df.groupby("citations")[labels].sum()
value_citations = value_citations.div(value_citations.sum(axis=1), axis=0)

citations_order = ["0-29", "10-29", "30-99", "60-99", "100+"]
mapping = {day: i for i, day in enumerate(citations_order)}
key = value_citations.index.map(mapping)
value_citations = value_citations.iloc[key.argsort()]
value_citations.index = value_citations.index.map(
    lambda x: f"{x}\n({totals[x]} papers)"
)

# stacked barplot
value_citations.plot(kind="area", stacked=True, rot=0, ax=ax2)

ax2.set_title("Values in influential NLP research")
ax2.set_xlabel("Citation count")
# ax2.set_ylabel('Funding percentages')

# show legend outside of plot
# ax = plt.gca()
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("img/values.png", dpi=300, bbox_inches="tight")

# plt.show()
