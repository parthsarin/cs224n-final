import pandas as pd
from json import load
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")

data = []
for file in glob("data/*.json"):
    slug = file.split("/")[-1]
    acl_id = slug[:-5]
    d = load(open(file))
    if "funding" not in d:
        continue
    if "values" not in d or not d["values"]:
        continue

    values = d["values"]
    espoused_values = list(values.keys())
    espoused_values = {k: 1 for k in espoused_values}

    funding = d["funding"]
    funding = {k: round(v) for k, v in funding.items()}
    d = {**funding, "acl_id": acl_id, **espoused_values}
    data.append(d)

df = pd.DataFrame(data).fillna(0)

funding = ["defense", "corporate", "research agency", "foundation", "none"]
values = [
    "building on past work",
    "novelty",
    "performance",
    "reproducibility",
    "ease of implementation",
    "fairness",
]

funding_values = []
for source in funding:
    total_funded_by_source = df[source].sum()
    X = df[df[source] == 1]
    num_papers = dict(X[values].sum())
    # if "ease of implementation" in num_papers:
    #     num_papers["implementability"] = num_papers.pop("ease of implementation")

    if source == 'research agency':
        source = 'agency'

    funding_values.append({"source": source, **num_papers})

funding_values = pd.DataFrame(funding_values)
df = funding_values.set_index("source")
df = df.apply(lambda x: x / sum(x), axis=1)
df.plot(kind="bar", stacked=True, rot=0)

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1, 0.5))

plt.title("What types of papers do different funders fund?")
plt.xlabel("Funding source")
plt.ylabel("Proportion of portfolio")

plt.tight_layout()
plt.savefig("img/funding_value.png", dpi=300, bbox_inches="tight")
