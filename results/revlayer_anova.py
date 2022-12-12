import pandas as pd
import scipy.stats as stats
import pylab

df = pd.read_csv("stat_comparison_results.csv",
                 index_col=0)

# Remove outliers to achieve normality
df = df.drop(0)

# Check normality assumption
print("---------------------------------------------------------------------------------------------------------------")
stats.probplot(df["Balanced"], dist="norm", plot=pylab)
print("\nShapiro-Wilk test on Balanced")
print(stats.shapiro(df["Balanced"]))
pylab.show()

stats.probplot(df["Unbalanced_less_params"], dist="norm", plot=pylab)
print("\nShapiro-Wilk test on Unbalanced_less_params")
print(stats.shapiro(df["Unbalanced_less_params"]))
pylab.show()

stats.probplot(df["Unbalanced_same_params"], dist="norm", plot=pylab)
print("\nShapiro-Wilk test on Unbalanced_same_params")
print(stats.shapiro(df["Unbalanced_same_params"]))
pylab.show()

# Check homoschedasticity assumption
print("---------------------------------------------------------------------------------------------------------------")
print(df.std())
print(stats.bartlett(df["Balanced"], df["Unbalanced_less_params"], df["Unbalanced_same_params"]))

# ANOVA
print("---------------------------------------------------------------------------------------------------------------")
print("\nMeans:")
print(df.mean())
print(stats.f_oneway(df["Balanced"], df["Unbalanced_less_params"], df["Unbalanced_same_params"], axis=0))

print("\nBalanced vs Unbalanced_less_params:")
print(stats.f_oneway(df["Balanced"], df["Unbalanced_less_params"], axis=0))

print("\nBalanced vs Unbalanced_same_params")
print(stats.f_oneway(df["Balanced"], df["Unbalanced_same_params"], axis=0))
