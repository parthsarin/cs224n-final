"""
File: parse_values.py
--------------------

This file contains the implementation of the parsing the values from the
output that Vyoma provided.
"""
from typing import Optional
import sys
import csv
import argparse
import re
import regex
from nltk import sent_tokenize
from tqdm import tqdm
from json import load, dump

csv.field_size_limit(sys.maxsize)

ARTICLE_RE = re.compile(
    r"I have the following research paper:\n(.*?)\nReturn a list of excerpts from this paper related to (.*?), which refers to",
    re.DOTALL,
)


def fuzzy_substring_search(
    major: str, minor: str, errs: int = 4
) -> Optional[regex.Match]:
    """
    Find the closest matching fuzzy substring.

    Args:
        major: the string to search in
        minor: the string to search with
        errs: the total number of errors

    Returns:
        Optional[regex.Match] object

    https://stackoverflow.com/questions/17740833/checking-fuzzy-approximate-substring-existing-in-a-longer-string-in-python
    """
    errs_ = 0
    minor = re.escape(minor)
    s = regex.search(f"({minor}){{e<={errs_}}}", major)
    while s is None and errs_ <= errs:
        errs_ += 1
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
    return s


def parse_values(article, response):
    out = []
    sentences = sent_tokenize(response)

    # for each sentence check if it's similar to the article
    # if it is, add it to the list of excerpts
    for sentence in sentences:
        if not sentence:
            continue
        match = fuzzy_substring_search(article, sentence, 4)
        if match:
            out.append(sentence.strip(" -*"))

    return out


def main(args):
    # get the papers and topics from the input file
    reader = csv.DictReader(open(args.input_file))
    n_rows = sum(1 for _ in reader)

    reader = csv.DictReader(open(args.input_file))
    for row in tqdm(reader, total=n_rows):
        acl_id = row["ACL_ID"]
        prompt = row["Prompt"]

        # use the regex to get the topic
        match = ARTICLE_RE.search(prompt)
        if match:
            article = match.group(1)
            topic = match.group(2)
        else:
            raise ValueError(f"Could not find a topic in the prompt: {prompt}")

        if not topic or not article:
            raise ValueError(
                f"Could not find a topic or article in the prompt: {prompt}"
            )

        values = parse_values(article, row["response"])

        if values:
            d = load(open(f"{args.out_folder}/{acl_id}.json"))
            d_vals = d.get("values", {})
            d_vals[topic] = values
            d["values"] = d_vals
            dump(d, open(f"{args.out_folder}/{acl_id}.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Topic modeling on the values that Vyoma's model produced."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The input file containing the values that Vyoma's model produced.",
    )
    parser.add_argument(
        "out_folder",
        type=str,
        help="The output folder to write the value data to.",
    )
    args = parser.parse_args()

    main(args)
