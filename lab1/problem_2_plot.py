# -*- coding: utf-8 -*-
"""
Use this script to generate plot pdfs
"""
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt


df = pd.read_csv("prob2_results/lab2_v0.csv", index_col=0)
df.columns = ['value function at the initial state']
ax = df.plot(legend=False)
ax.set_xlabel("discount factor (λ)")
ax.set_ylabel("Value Function at the initial state")
plt.savefig("report/prob2_vfunc.pdf", bbox_inches='tight')
