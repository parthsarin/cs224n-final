"""
File: csv_to_json.py
--------------------

Parses the CSV file into separate JSON files so we can access the paper data a
bit more easily.
"""
import argparse
from csv import DictReader, reader
from json import load, dump
from tqdm import tqdm
import sys
import csv

csv.field_size_limit(sys.maxsize)


def main(args):
    n_rows = sum(1 for row in reader(open(args.csv_file, 'r'))) - 1
    f = DictReader(open(args.csv_file, 'r'))

    for row in tqdm(f, total=n_rows):
        acl_id = row.pop('acl_id')
        filename = f"data/{acl_id}.json"

        # load existing data, if any
        try:
            d = load(open(filename, 'r'))
        except FileNotFoundError:
            d = {}

        # isolate article from the prompt
        d = {
            **d,
            'article': row.pop('full_text'),
            'abstract': row.pop('abstract'),
            'countries': eval(row.pop('countries')),
            'languages': eval(row.pop('final_langs')),
            'numcitedby': row.pop('numcitedby'),
            'year': row.pop('year'),
            'month': row.pop('month'),
            'title': row.pop('title'),
        }

        # write the data back to the file
        dump(d, open(f'{args.output_dir}/{acl_id}.json', 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parse the CSV file into separate JSON files'
    )

    parser.add_argument(
        'csv_file',
        type=str,
        help='the CSV file to parse'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='the directory to output the JSON files'
    )

    args = parser.parse_args()
    main(args)
