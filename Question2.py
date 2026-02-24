import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('crime1 (1).csv')

column = df["ViolentCrimesPerPop"]

plt.figure()
plt.hist(column, bins = 20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.boxplot(column)
plt.title("Boxplot of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Value")
plt.show()


#We already know from question 1 that the histogram should be right skewed( using mean and median values and comparing them) - and the graph shows that, we can see that the values are more concentrated on the lower
#end therefor showing that fewer communities have very high crime rates.
#The box plot shows the median clearly as a line inside the box.
#The median is slightly lower than the mean confirming the right skewness.
#The box plot does not but could show points beyond the whiskers which tells us that there are most likely no outliers in the higher range.
