# -*- coding: utf-8 -*-
"""
Use this script to generate plot pdfs
"""
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt


class Config:
    outfolder = "./lab3_csv"
    epsilons = [
        0.1,
        0.2,
        0.3,
        0.01,
        0.05,
        # 0.005,
    ]


cfg = Config

sassa_v_funs = dict()
sassa_norm_delta_q = dict()

for eps in cfg.epsilons:
    csvfile = "SARSA-eps-{:.0E}.csv".format(eps)
    df = pd.read_csv(join(cfg.outfolder, csvfile), index_col=0)
    col = "ε={:.0E}".format(eps)
    sassa_v_funs[col] = df['v_fun_0']
    sassa_norm_delta_q[col] = df['norm_delta_q']


sassa_v_fun_df = pd.DataFrame(sassa_v_funs)
ax = sassa_v_fun_df.plot(figsize=(10, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Time step")
ax.set_ylabel("Value Function")
# ax.set_title("SASSA: Value Function over time at the inital state")
plt.savefig("report/sassa_v_fun_0.pdf", bbox_inches='tight')
plt.cla()
# # ---------------------------------------------------------------
sassa_norm_delta_q_df = pd.DataFrame(sassa_norm_delta_q)
ax = sassa_norm_delta_q_df.plot(figsize=(10, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Time step")
ax.set_ylabel("Frobenius norm of delta Q")
plt.savefig("report/sassa_norm_delta_q.pdf", bbox_inches='tight')
plt.cla()

# log + rolling
sassa_norm_delta_q_df = sassa_norm_delta_q_df.rolling(window=20).mean()
ax = sassa_norm_delta_q_df.plot(figsize=(10, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Time step")
ax.set_ylabel("Frobenius norm of delta Q in log scale")
ax.set_yscale('log')
plt.savefig("report/sassa_norm_delta_q_log.pdf", bbox_inches='tight')
plt.cla()

# -----------------------------------------------------------------------------
# Q function plots
csvfile = "Q_learn.csv"
df = pd.read_csv(join(cfg.outfolder, csvfile), index_col=0)
# Value function at initial state
ax = df['v_fun_0'].plot(figsize=(10, 6))
ax.set_xlabel("Time step")
ax.set_ylabel("Value Function")
plt.savefig("report/q_learn_v_fun_0.pdf", bbox_inches='tight')
plt.cla()

s = df['norm_delta_q']
ax = s.plot(figsize=(10, 6))
ax.set_xlabel("Time step")
ax.set_ylabel("Frobenius norm of delta Q.")
plt.savefig("report/q_learn_norm_delta_q.pdf", bbox_inches='tight')
plt.cla()


s = s.rolling(window=20).mean()
ax = s.plot(figsize=(10, 6))
ax.set_xlabel("Time step")
ax.set_ylabel("Frobenius norm of delta Q in log scale")
ax.set_yscale('log')
plt.savefig("report/q_learn_norm_delta_q_log.pdf", bbox_inches='tight')
