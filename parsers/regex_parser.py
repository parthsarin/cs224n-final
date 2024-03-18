# -*- coding: utf-8 -*-
"""Regex Parser.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dDB83ktRolvrW75B7Ucr_fuHlxn465x8
"""

import re
import json

orgs = {
    "darpa": "military",
    "us defense advanced research projects agency": "military",
    "u.s. department of homeland security": "military",
    "dhs": "military",
    "office of naval research": "military",
    "dod": "military",
    "department of defense": "military",

    "tencent": "corporate",
    "google": "corporate",
    "nec labs america": "corporate",
    "abridge": "corporate",
    "vinai": "corporate",
    "at&t": "corporate",
    "adobe": "corporate",
    "ntt": "corporate",
    "china mobile": "corporate",

    "nsf": "research agency",
    "natural science foundation of china": "research agency",
    "german research foundation": "research agency",
    "dfg": "research agency",
    "european union": "research agency",
    "science foundation ireland": "research agency",
    "jsps": "research agency",
    "eu": "research agency",
    "french national agency for research": "research agency",
    "nar": "research agency",
    "european commission’s 7th framework program": "research agency",
    "german": "research agency",
    "polish ministry of science and higher education": "research agency",
    "ic3": "research agency",
    "national science council": "research agency",
    "science foundation": "research agency",


    "asahi glass foundation": "foundation",
}

def classify_paper(paper_text):
  paper = paper_text.lower()

  # 1. get the final section of the paper
  slice_indexes = [
      paper.find("acknowledgement"),
      paper.find("acknowledgements"),
      paper.find("acknowledgments"),
      paper.find("acknowledgment"),
      paper.find("funding"),
      paper.find("competing interest"),
  ]

  slice_indexes = [x for x in slice_indexes if x != -1]
  if slice_indexes:
    paper = paper[min(slice_indexes):]

  funding = []
  for source in orgs:
      if re.search(rf"\b{source}\b", paper):
          funding.append(orgs[source])

  return list(set(funding))

orgs_lst = orgs.keys()

regex = r"acknowledg[e]?ments?(?:(?!\backnowledg[e]?ments?\b).){0,200}?\b(?P<org_name>" + '|'.join(map(re.escape, orgs_lst)) + r")\b"

file_path = '2020.emnlp-main.555.json'

# Read the JSON file
with open(file_path, 'r') as file:
  data = json.load(file)
  article_text = data['article'].lower()
  funding = classify_paper(article_text)
  print(funding)
  # match = re.search(regex, article_text)
  # if match:
  #   # print(f"Found organization: {match.group('org_name')}")
  #   org = match.group('org_name')
  #   print(orgs[org])
  # else:
  #   print("No matching organization found.")

# Get list of filenames
filenames = open("testset_filenames.csv", 'r')
lines = filenames.readlines()
testset_filenames = []
for line in lines:
  testset_filenames.append(line.strip())

testset_filenames

regex_results_dict = {}
for test_article in testset_filenames:
  filename = test_article + ".json"
  with open(filename, 'r') as file:
    data = json.load(file)
    article_text = data['article'].lower()
    # print(article_text)
    funding = classify_paper(article_text)
    if not funding:
      funding = [0, 0, 0, 0, 1]
    else:
      labels = ['military', 'corporate', 'research_agency', 'foundation']
      funding = [int(x in funding) for x in labels] + [0]
    regex_results_dict[filename[:-5]] = funding

regex_results_dict

len(regex_results_dict.keys())

import csv
import pandas as pd
test_labels = pd.read_csv("testset.csv")

test_labels

testset_labels_dict = dict(zip(test_labels.file_id, test_labels[['military', 'corporate', 'research_agency', 'foundation', 'none']].to_numpy()))

len(testset_labels_dict.keys())

total_correct = 0
for key in testset_labels_dict.keys():
  labels = testset_labels_dict[key]
  preds = regex_results_dict[key]
  acc_vector = 1- (preds + labels) % 2
  total_correct += sum(acc_vector)
print(f' test accuracy: {total_correct / (5 * 30):.4f}', end='')

import csv
import pandas as pd
results = pd.read_csv('paper funding labels - Sheet8.csv')

import csv
training_data = []
with open('paper funding labels - Sheet9.csv') as file_obj:
  reader_obj = csv.reader(file_obj)
  for row in reader_obj:
    training_data.append(row[0])

# Open file
test_data = []
test_data_options = []
with open('paper funding labels - Sheet8.csv') as file_obj:

    # Create reader object by passing the file
    # object to reader method
    reader_obj = csv.reader(file_obj)

    # Iterate over each row in the csv
    # file using reader object
    counter = 0
    to_print = [i]
    for row in reader_obj:
      test_data_options.append(row[0])

import random
random.shuffle(test_data_options)

for option in test_data_options:
  if option not in training_data:
    test_data.append(option)
  if len(test_data) == 60:
    break

test_data

for data in test_data:
  print(data[:-5])

for title in to_print:
  print(title)

len(to_print)
