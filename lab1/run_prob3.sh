#!/bin/bash

mkdir -p lab3_csv

python problem_3.py --algo SARSA --eps 0.1  --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.3  --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.5  --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.7  --outfolder=lab3_csv
python problem_3.py --algo SARSA --eps 0.9  --outfolder=lab3_csv
python problem_3.py --algo Q_learn  --outfolder=lab3_csv
