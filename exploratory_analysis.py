import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv('framingham.csv')

# Create a directory for saving plots
import os
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# 1. Basic Statistics
print("Basic Statistics of Numerical Features:")
print(df.describe())

# 2. Target Variable Distribution
plt.figure(figsize=(8, 6))
plt.hist(df['TenYearCHD'], bins=2, rwidth=0.8)
plt.title('Distribution of 10 Year CHD Risk')
plt.xlabel('CHD Risk (0: No, 1: Yes)')
plt.ylabel('Count')
plt.savefig('eda_plots/target_distribution.png')
plt.close()

# 3. Age Distribution by CHD Risk
plt.figure(figsize=(10, 6))
plt.boxplot([df[df['TenYearCHD']==0]['age'], df[df['TenYearCHD']==1]['age']], 
            labels=['No CHD', 'Has CHD'])
plt.title('Age Distribution by CHD Risk')
plt.ylabel('Age')
plt.savefig('eda_plots/age_distribution.png')
plt.close()

# 4. Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('eda_plots/correlation_matrix.png')
plt.close()

# 5. Distribution of Key Numerical Features
numerical_features = ['age', 'sysBP', 'BMI', 'glucose']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df[col].dropna(), bins=30)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_plots/numerical_distributions.png')
plt.close()

# 6. Risk Factors Analysis
categorical_features = ['male', 'currentSmoker', 'prevalentHyp', 'diabetes']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    risk_by_factor = df.groupby(col)['TenYearCHD'].mean()
    axes[idx].bar(risk_by_factor.index, risk_by_factor.values)
    axes[idx].set_title(f'CHD Risk by {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Risk Probability')

plt.tight_layout()
plt.savefig('eda_plots/risk_factors.png')
plt.close()

# 7. Statistical Tests
print("\nStatistical Tests (comparing groups with and without CHD):")
numerical_features = ['age', 'sysBP', 'BMI', 'glucose', 'totChol', 'heartRate']
for col in numerical_features:
    stat, p_value = stats.ttest_ind(
        df[df['TenYearCHD'] == 1][col].dropna(),
        df[df['TenYearCHD'] == 0][col].dropna()
    )
    print(f"\n{col}:")
    print(f"t-statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")

# 8. Missing Values Analysis
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100

print("\nMissing Values Analysis:")
for column, percentage in missing_percentages.items():
    print(f"{column}: {percentage:.2f}% missing")

# 9. Summary Statistics by CHD Risk
print("\nSummary Statistics by CHD Risk:")
summary_stats = df.groupby('TenYearCHD').agg({
    'age': ['mean', 'std'],
    'sysBP': ['mean', 'std'],
    'BMI': ['mean', 'std'],
    'glucose': ['mean', 'std']
}).round(2)

# Format the output for better readability
print("\nFor patients WITHOUT CHD (0):")
print(summary_stats.loc[0])
print("\nFor patients WITH CHD (1):")
print(summary_stats.loc[1])

print("\nEDA completed! Plots have been saved in the 'eda_plots' directory.") 