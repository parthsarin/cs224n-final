"""
File: inject_samples.py
-----------------------

Injects positive and negative samples into the specified dataset.
"""
import argparse
import pandas as pd
from glob import glob
from nltk import sent_tokenize
import random
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def inject_sample(sample, original):
    sentences = sent_tokenize(original)

    # replace a random sentence with the sample
    i = random.randint(0, len(sentences) - 1)
    sentences[i] = sample

    # check to make sure it's less than 512 tokens
    proposal = " ".join(sentences)
    proposal_len = len(tokenizer.encode(proposal, add_special_tokens=False))

    # while it's too long, trim sentences from either end, making sure not to trim the sample
    while proposal_len > 512:
        if i < len(sentences) - 1:
            sentences = sentences[:-1]
        else:
            sentences = sentences[1:]
        proposal = " ".join(sentences)
        proposal_len = len(tokenizer.encode(proposal, add_special_tokens=False))

    return proposal


def main(args):
    # load the dataset
    df = pd.read_csv(args.dataset, dtype={"chunk_id": str})

    pos_categories = [
        f.split("/")[-1].split(".")[0] for f in glob(f"{args.pos_dir}/*.txt")
    ]
    neg_categories = [
        f.split("/")[-1].split(".")[0] for f in glob(f"{args.neg_dir}/*.txt")
    ]
    pos = [open(f).read().strip().split("\n") for f in glob(f"{args.pos_dir}/*.txt")]
    neg = [open(f).read().strip().split("\n") for f in glob(f"{args.neg_dir}/*.txt")]

    # inject the positive samples
    for category, samples in zip(pos_categories, pos):
        for sample in samples:
            k = args.pos_branching_factor

            # sample k indices with none = true
            x = df[df["none"] == 1]
            indices = x.sample(k).index
            x = df.loc[indices]

            # inject the samples
            for i, row in x.iterrows():
                new_text = inject_sample(sample, row["text"])

                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.loc[len(df) - 1, "text"] = new_text
                df.loc[len(df) - 1, "chunk_id"] += f"-pos-{category}-{i}"
                df.loc[len(df) - 1, "none"] = False
                df.loc[len(df) - 1, category] = True

    # inject the negative samples
    for category, samples in zip(neg_categories, neg):
        for sample in samples:
            k = args.neg_branching_factor

            # sample k indices with none = true
            x = df[df["none"] == 1]
            indices = x.sample(k).index
            x = df.loc[indices]

            # inject the samples
            for i, row in x.iterrows():
                new_text = inject_sample(sample, row["text"])

                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.loc[len(df) - 1, "text"] = new_text
                df.loc[len(df) - 1, "chunk_id"] += f"-neg-{category}-{i}"
                # original funding hasn't changed

    # write the dataset to a file
    df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject positive and negative samples into a specified dataset"
    )
    parser.add_argument("dataset", type=str, help="The dataset to inject samples into")
    parser.add_argument(
        "pos_dir", type=str, help="The directory containing the positive samples"
    )
    parser.add_argument(
        "neg_dir", type=str, help="The directory containing the negative samples"
    )
    parser.add_argument(
        "out_file", type=str, help="The output file to write the dataset to"
    )
    parser.add_argument(
        "-nf",
        "--neg-branching-factor",
        type=int,
        default=3,
        help="The number of data points to perturb for each negative sample.",
    )
    parser.add_argument(
        "-pf",
        "--pos-branching-factor",
        type=int,
        default=8,
        help="The number of data points to perturb for each positive sample.",
    )

    args = parser.parse_args()
    main(args)
