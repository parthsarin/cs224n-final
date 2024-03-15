"""
File: topic_model.py
--------------------

This file contains the implementation of the topic model that we will use to
classify the articles into different categories.
"""
from bertopic import BERTopic
from typing import List
import argparse
from glob import glob
from json import load
from collections import defaultdict
import pandas as pd


def fit_model_to_topic(docs: List[str]):
    m = BERTopic()
    topics, probs = m.fit_transform(docs)
    return topics, probs, m


def main(args):
    # get the topic data
    topic_data = defaultdict(dict)
    for filename in glob(f"{args.in_dir}/*.json"):
        data = load(open(filename))
        stem = filename.split("/")[-1]
        acl_id = stem[: stem.rfind(".json")]
        values = data.get("values", {})

        for value, snippet in values.items():
            topic_data[value][acl_id] = snippet

    print("loaded topic data")

    # fit the model to the topic data
    for topic in topic_data:
        print(f"fitting model to topic {topic}...")
        acl_ids = list(topic_data[topic].keys())
        docs = list(topic_data[topic].values())
        topics, probs, model = fit_model_to_topic(docs)

        topic_df = pd.DataFrame(
            {"acl_id": acl_ids, "topic": topics, "probability": probs}
        )
        topic_df.to_csv(f"{args.out_dir}/{topic}.csv", index=False)
        model.get_document_info().to_csv(
            f"{args.out_dir}/{topic}_info.csv", index=False
        )

        print(f"finished fitting model to topic {topic}, wrote to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_dir",
        type=str,
        help="The input directory containing the articles to classify.",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="The output directory to write the topic models to.",
    )

    args = parser.parse_args()
    main(args)
