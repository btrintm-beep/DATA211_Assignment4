import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('crime1 (1).csv')

column = df["ViolentCrimesPerPop"]

mean_value = column.mean()
median_value = column.median()
std_value = column.std()
min_value = column.min()
max_value = column.max()

print("Mean Value: ", mean_value)
print("Median Value: ", median_value)
print("Std Value: ", std_value)
print("Min Value: ", min_value)
print("Max Value: ", max_value)

plt.hist(column)
plt.show()

#The mean value is larger than the median value and we can
#see that the graph is right skewed

#outliers effect the mean more than the median as the mean uses an average of every number which the outlier would skew
#while the median goes through the middle value which even with an addition of outliers should not change it to much since its so
#outweighed by the rest.

