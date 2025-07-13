import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models and their corresponding rank variability values across the metrics
models = ['REL_POFHG', 'EHCF', 'SELFCFHE LGN', 'LRGCN', 'LightGCN', 'FSLR', 'BNSLIM-ADM', 'FAIRMF']
metrics = ['NDCG', 'Precision', 'Recall', 'Hit Ratio']

rank_variability = {
    'REL_POFHG': [1.81, 1.88, 1.85, 1.86],
    'EHCF': [1.73, 1.88, 1.85, 1.87],
    'SELFCFHE LGN': [1.79, 1.85, 1.79, 1.81],
    'LRGCN': [1.72, 1.83, 1.76, 1.80],
    'LightGCN': [1.69, 1.76, 1.74, 1.77],
    'FSLR': [1.76, 1.74, 1.79, 1.75],
    'BNSLIM-ADM': [1.87, 1.84, 1.84, 1.84],
    'FAIRMF': [1.84, 1.84, 1.84, 1.84]
}



# --- 2. Heatmap Representation ---

# Convert the rank variability data into a pandas DataFrame
data = {
    'NDCG': [1.81, 1.73, 1.79, 1.72, 1.69, 1.76, 1.87, 1.84],
    'Precision': [1.88, 1.88, 1.85, 1.83, 1.76, 1.74, 1.84, 1.84],
    'Recall': [1.85, 1.85, 1.79, 1.76, 1.74, 1.79, 1.84, 1.84],
    'Hit Ratio': [1.86, 1.87, 1.81, 1.80, 1.77, 1.75, 1.84, 1.84]
}

# Create a DataFrame
df = pd.DataFrame(data, index=models)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Rank Variability'}, fmt='.2f')

# Title and labels
#plt.title('Rank Variability Heatmap Across Models and Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')

# Display the heatmap
plt.tight_layout()
plt.show()


data = {
    'NDCG': {
        'REL_POFHG': [0.204, 0.084, 0.411],
        'EHCF': [0.0925, 0.0913, 0.987],
        'SELFCFHE LGN': [0.0959, 0.0834, 0.870],
        'LRGCN': [0.1336, 0.0964, 0.722],
        'LightGCN': [0.1597, 0.0747, 0.467],
        'FSLR': [0.2203, 0.2115, 0.960],
        'BNSLIM-ADM': [0.252, 0.2687, 1.067],
        'FAIRMF': [0.0708, 0.0573, 0.810]
    },
    'Precision': {
        'REL_POFHG': [0.319, 0.101, 0.317],
        'EHCF': [0.2419, 0.1758, 0.727],
        'SELFCFHE LGN': [0.2117, 0.1826, 0.863],
        'LRGCN': [0.2594, 0.1696, 0.653],
        'LightGCN': [0.387, 0.1147, 0.296],
        'FSLR': [0.4866, 0.317, 0.651],
        'BNSLIM-ADM': [0.427, 0.3589, 0.841],
        'FAIRMF': [0.2292, 0.1603, 0.699]
    },
    'Recall': {
        'REL_POFHG': [0.181, 0.087, 0.481],
        'EHCF': [0.0367, 0.0274, 0.747],
        'SELFCFHE LGN': [0.0393, 0.0245, 0.623],
        'LRGCN': [0.0505, 0.0223, 0.442],
        'LightGCN': [0.0632, 0.0127, 0.201],
        'FSLR': [0.0836, 0.0356, 0.426],
        'BNSLIM-ADM': [0.0704, 0.0336, 0.477],
        'FAIRMF': [0.0325, 0.0223, 0.685]
    },
    'Hit Ratio': {
        'REL_POFHG': [0.0515, 0.048, 0.932],
        'EHCF': [0.1748, 0.1938, 1.109],
        'SELFCFHE LGN': [0.2009, 0.1609, 0.802],
        'LRGCN': [0.2579, 0.1673, 0.649],
        'LightGCN': [0.2252, 0.1449, 0.644],
        'FSLR': [0.2498, 0.1875, 0.751],
        'BNSLIM-ADM': [0.2799, 0.2387, 0.852],
        'FAIRMF': [0.1057, 0.0785, 0.743]
    }
}

# Convert the data into a DataFrame for easy manipulation
df = pd.DataFrame(data)

# Separate out the Mean, Std. Dev., and CV into their own DataFrames
mean_df = pd.DataFrame({metric: [values[0] for values in model_values.values()] for metric, model_values in data.items()}, index=data['NDCG'].keys())
std_df = pd.DataFrame({metric: [values[1] for values in model_values.values()] for metric, model_values in data.items()}, index=data['NDCG'].keys())
cv_df = pd.DataFrame({metric: [values[2] for values in model_values.values()] for metric, model_values in data.items()}, index=data['NDCG'].keys())

# Plot heatmap for **Mean**
plt.figure(figsize=(10, 6))
sns.heatmap(mean_df, annot=True, cmap='coolwarm', fmt='.3f', cbar_kws={'label': 'Mean Value'})
plt.title('Mean Stability Analysis Across Models and Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')
plt.tight_layout()
plt.savefig("stability_mean", dpi=300, bbox_inches="tight")
plt.show()

# Plot heatmap for **Standard Deviation (Std. Dev.)**
plt.figure(figsize=(10, 6))
sns.heatmap(std_df, annot=True, cmap='coolwarm', fmt='.3f', cbar_kws={'label': 'Standard Deviation'})
plt.title('Standard Deviation Stability Analysis Across Models and Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')
plt.tight_layout()
plt.savefig("stability_std", dpi=300, bbox_inches="tight")
plt.show()

# Plot heatmap for **Coefficient of Variation (CV)**
plt.figure(figsize=(10, 6))
sns.heatmap(cv_df, annot=True, cmap='coolwarm', fmt='.3f', cbar_kws={'label': 'Coefficient of Variation (CV)'})
plt.title('Coefficient of Variation Stability Analysis Across Models and Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')
plt.tight_layout()
plt.savefig("stability_cv", dpi=300, bbox_inches="tight")
plt.show()
