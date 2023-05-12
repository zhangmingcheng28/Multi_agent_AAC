# -*- coding: utf-8 -*-
"""
@Time    : 5/12/2023 12:56 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')
fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)
fig3, ax3 = plt.subplots(1, 1)
fig4, ax4 = plt.subplots(1, 1)
fig5, ax5 = plt.subplots(1, 1)
fig6, ax6 = plt.subplots(1, 1)
fig7, ax7 = plt.subplots(1, 1)
fig8, ax8 = plt.subplots(1, 1)

df_training = pd.read_csv(r'F:\githubClone\Multi_agent_AAC\titanic.csv')
threshold = len(df_training) * 0.5
# Drop columns with more than 50% missing values
df_clean = df_training.dropna(axis=1, thresh=threshold)  # Q1a, cabin column dropped

average_survival_rate = round(((df_clean['Survived'] == 1).sum() / len(df_training)) * 100, 2)  # Q1b in 2dp

# Q1c
ax1.pie(df_clean['Sex'].value_counts(), labels=df_clean['Sex'].value_counts().index, autopct='%1.1f%%')
ax1.set_title('Pie chart of Sex')
ax2.pie(df_clean['Pclass'].value_counts(), labels=df_clean['Pclass'].value_counts().index, autopct='%1.1f%%')
ax2.set_title('Pie chart of Pclass')

passener_alone = sum((df_clean['SibSp'] == 0) & (df_clean['Parch'] == 0))  # Q1d

ax3.hist(df_clean['Fare'])  # Q1e
# Insight: The fare data has a right skewed distribution. Meaning most of the passenger choose a normal cabin (lower price), which is reasonable, not every one will buy first class ticket in a plane.
ax3.set_title('Data distribution of Fare column in data frame')
ax4.boxplot(df_clean['Fare'], vert=True)
ax4.set_title('Fare column box plot')  # Q1f, outlier are market in using scatter points

ax5.hist(df_clean['Age'])  # Q1g
# Insight: The fare data can be a approximate normal distribution. Indicates that most of the passengers are around 20~40, normally they are the people with higher purchasing power
ax5.set_title('Data distribution of Age column in data frame')
ax6.boxplot(df_clean['Age'].dropna(), vert=True)
ax6.set_title('Age column box plot')

# Calculate the Pearson correlation coefficient between the two columns
correlation = df_clean['Pclass'].corr(df_clean['Survived'])  # Q1h, correlation is just -0.338, so they don't have much correlations


# Q1j
# Define the age bands
bins = [0, 16, 32, 48, 64, 80]
# Define the labels for the age bands
labels = [0, 1, 2, 3, 4]
# Create the 'AgeBand' column by cutting the 'Age' column into bins
AgeBand = pd.cut(df_clean['Age'].dropna(), bins, labels=labels, right=False, include_lowest=True)

# Q1k
# remove row, when age has nan
df_1k = pd.concat([df_clean['Survived'], df_clean['Age']], axis=1)
df_1k_clean = df_1k.dropna(subset=['Age'])
bins = [0, 16, 32, 48, 64, 80]
labels = [0, 1, 2, 3, 4]
AgeBand_1k = pd.cut(df_1k_clean['Age'], bins, labels=labels, right=False, include_lowest=True)

# Calculate the Pearson correlation coefficient between the two columns
correlation_1k = AgeBand_1k.corr(df_1k_clean['Survived'])   # Q1k, correlation is just -0.066, compare to "Pclass" and "Survived", this two seems more related

# Q1l
df_clean['Title'] = df_clean['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
Uncommon = ["Rev", "Major", "Col", "Don", "Sir", "Capt", "Jonkheer", "Lady", "Mme", "Countess"]
Miss = ["Mlle", "Ms"]
display_dict = {}
Miss_Count = 0
Uncommon_Count = 0
for i in df_clean['Title'].value_counts().items():
    if i[0] in Miss:
        Miss_Count = Miss_Count + i[1]
        display_dict["Miss"] = Miss_Count
    elif i[0] in Uncommon:
        Uncommon_Count = Uncommon_Count + i[1]
        display_dict["Uncommon"] = Uncommon_Count
    else:
        display_dict[i[0]] = i[1]

# Q1m
df_clean['Alone'] = np.where((df_clean['SibSp'] == 0) & (df_clean['Parch'] == 0), 1, 0)
correlation_1m = df_clean['Alone'].corr(df_clean['Survived'])  # -0.203, -ve value, so no correlation

# Q1n
df_clean['TravelWith'] = df_clean['SibSp'] + df_clean['Parch']
correlation_1n = df_clean['TravelWith'].corr(df_clean['Survived'])  # 0.0166, not much correlation

# Q1o
LogFare = np.log(df_clean['Fare'] + 1)
# Divide the 'LogFare' column into 5 even bands
labels = [0, 1, 2, 3, 4]
df_clean['LogFareBand'] = pd.cut(LogFare, bins=5, labels=labels)
correlation_1o = df_clean['LogFareBand'].corr(df_clean['Survived'])  # 0.33, a positive Pearson Correlation, so, have some degree of relationship.

# Q1p(i)

# use df_1k_clean, so that Nan age is removed







plt.show()
print(df_training.head())
print("done")