"""
File: json_to_csv.py
--------------------

Parses the JSON files into a single CSV file containing the ACL ID and funding
sources (if needed)
"""
from glob import glob
from json import load
from tqdm import tqdm
import pandas as pd
import argparse


def main(args):
    out = []
    for path in tqdm(glob(f'{args.in_dir}/*.json')):
        data = load(open(path, 'r'))
        filename = path.split('/')[-1]
        acl_id = filename[:filename.rfind('.')]
        try:
            funding = data['funding']
        except KeyError:
            tqdm.write(f'No funding found in file {path}')

        if args.round > 0:
            funding = {k: round(v, args.round) for k, v in funding.items()}

        out.append({
            'acl_id': acl_id,
            **funding
        })

    df = pd.DataFrame(out)
    df.to_csv(args.csv_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parse the JSON files into a single CSV file'
    )

    parser.add_argument(
        'in_dir',
        type=str,
        help='the directory containing the JSON files'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='the CSV file to output'
    )
    parser.add_argument(
        '-r', '--round',
        type=int,
        default=-1,
        help='the number of decimal places to round the funding sources to'
    )

    args = parser.parse_args()
    main(args)
