"""
File: topic_model.py
--------------------

This file contains the implementation of the topic modeling on the values that
Vyoma's model produced.
"""
from bertopic import BERTopic
from typing import List


def fit_model_to_topic(docs: List[str]):
    return BERTopic().fit_transform(docs)
