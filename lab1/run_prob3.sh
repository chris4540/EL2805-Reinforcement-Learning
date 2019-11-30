#!/bin/bash
set -e

mkdir -p lab3_csv

python problem_3.py --algo SARSA --eps 0.1   --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.2   --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.3   --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.01  --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.05  --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.005 --outfolder=lab3_csv
python problem_3.py --algo Q_learn  --outfolder=lab3_csv
