import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
train_df = pd.read_csv('train.csv')

# 2. DATA CLEANING
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# 3. CREATE THE SINGLE WINDOW DASHBOARD
# We create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Titanic Data Analysis Dashboard', fontsize=20)

# Plot 1: Survival by Gender
sns.barplot(ax=axes[0], x='Sex', y='Survived', data=train_df, palette='magma')
axes[0].set_title('Survival Rate by Gender')

# Plot 2: Survival by Class
sns.countplot(ax=axes[1], x='Pclass', hue='Survived', data=train_df, palette='viridis')
axes[1].set_title('Survival Count by Class')

# Plot 3: Heatmap of Correlations
numeric_cols = train_df.select_dtypes(include=[np.number])
sns.heatmap(ax=axes[2], data=numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
axes[2].set_title('Variable Correlation Heatmap')

# Adjust layout so labels don't overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# SHOW THE SINGLE WINDOW
plt.show()

print("Execution Complete: One window displayed with all 3 plots.")