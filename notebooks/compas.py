import matplotlib.pyplot as plt
import pandas as pd

from fairlens.bias.viz import plt_group_dist

df = pd.read_csv("datasets/compas-scores-raw.csv")

g1 = {"Ethnic_Code_Text": ["African-American", "African-Am"]}
g2 = {"Ethnic_Code_Text": ["Caucasian"]}

plt_group_dist(df, g1, g2, "RawScore")
plt.savefig("plot.png")
