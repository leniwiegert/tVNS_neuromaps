import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import ptitprince as pt
from scipy.stats import pearsonr

## IMPORTS

data_directory = '/home/leni/Documents/Master/data/'

# MEANS
# Load numpy array from the file in data_directory
file_path_cort_mean = os.path.join(data_directory, 'corr_values_mean_cortical.npy')
corr_values_mean_cortical = np.load(file_path_cort_mean)
print(corr_values_mean_cortical.shape)

file_path_subcort_mean = os.path.join(data_directory, 'corr_values_mean_subcortical.npy')
corr_values_mean_subcortical = np.load(file_path_subcort_mean)
print(corr_values_mean_subcortical.shape)

# INDIVIDUAL LEVEL

# Load numpy array from the file in data_directory
file_path_cort = os.path.join(data_directory, 'corr_values_cortical_single_maps.npy')
corr_values_ss_cortical_array = np.load(file_path_cort, allow_pickle=True).item()
print(len(corr_values_ss_cortical_array))

file_path_subcort = os.path.join(data_directory, 'corr_values_subcortical_single_maps.npy')
corr_values_ss_subcortical_array = np.load(file_path_subcort,  allow_pickle=True).item()
print(len(corr_values_ss_subcortical_array))


'''
# MEAN DATA PLOTTING

x_values = ['Cortical', 'Subcortical']

# Define color categories for brain maps
color_categories = {
    0: [0, 1],
    1: [2],
    2: [3, 4, 5, 6],
    3: [7],
    4: [8],
    5: [9],
    6: [10]
}

# Colors for the color categories
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

# Plotting spaghetti plot
plt.figure(figsize=(12, 6))

for category, color in zip(color_categories.values(), colors):
    for i in category:
        y_values = [corr_values_mean_cortical[i], corr_values_mean_subcortical[i]]
        plt.plot(x_values, y_values, marker='o', color=color, label=f'Map {i+1}')

# Customize plot
plt.title('Spaghetti Plot of r-values')
plt.xlabel('Category')
plt.ylabel('r-values')
plt.grid(True)
plt.legend(title='Brain Maps', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Raincloud plot
plt.figure(figsize=(12, 6))

data = corr_values_mean_cortical + corr_values_mean_subcortical
categories = ['Cortical'] * len(corr_values_mean_cortical) + ['Subcortical'] * len(corr_values_mean_subcortical)

df = pd.DataFrame({'Data': data, 'Category': categories})

pt.RainCloud(x='Category', y='Data', data=df, palette='Set2',
             width_viol=0.8, width_box=0.4, alpha=0.7, dodge=True)
plt.title('Raincloud Plot of r-values', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('r-values', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

'''

# INDIVIDUAL DATA PLOTTING
'''
# Print loaded data structures
print("Cortical data:")
print(corr_values_ss_cortical_array)
print("\nSubcortical data:")


# Iterate over maps and print lengths of cortical and subcortical values
for map_name, cortical_values in corr_values_ss_cortical_array.items():
    subcortical_values = corr_values_ss_subcortical_array.get(map_name, None)
    print(f"\nMap: {map_name}")
    print(f"Cortical length: {len(cortical_values)}")
    print(f"Subcortical length: {len(subcortical_values) if subcortical_values is not None else 'N/A'}")


# Plot raincloud plots for each map
sns.set(style="whitegrid", palette="pastel")

for map_name, cortical_values in corr_values_ss_cortical_array.items():
    plt.figure(figsize=(8, 6))  # Create a new figure for each brain map

    subcortical_values = corr_values_ss_subcortical_array.get(map_name, None)

    # Check if subcortical_values is None or has different length
    if subcortical_values is None or len(subcortical_values) != len(cortical_values):
        print(f"Skipping map {map_name}: subcortical values not available or different length.")
        continue

    # Create DataFrame for raincloud plot
    df = pd.DataFrame({
        'Category': ['Cortical'] * len(cortical_values) + ['Subcortical'] * len(subcortical_values),
        'Value': cortical_values + subcortical_values
    })

# commented out for correlation calulcation
    # Plot raincloud plot
    sns.violinplot(x='Category', y='Value', data=df, inner="quartile", palette=["lightblue", "lightgreen"])
    sns.stripplot(x='Category', y='Value', data=df, jitter=True, size=3, color="gray", alpha=0.5)

    # Customize plot
    plt.title(map_name)
    plt.xlabel('Category')
    plt.ylabel('Correlation Value')
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()
'''


# Create an empty list to store combined data
combined_data = []

# Combine cortical and subcortical values into a single DataFrame
for map_name, cortical_values in corr_values_ss_cortical_array.items():
    subcortical_values = corr_values_ss_subcortical_array.get(map_name, None)
    if subcortical_values is not None and len(subcortical_values) == len(cortical_values):
        combined_data.extend([(map_name, 'Cortical', value) for value in cortical_values])
        combined_data.extend([(map_name, 'Subcortical', value) for value in subcortical_values])

# Create DataFrame from combined data
df = pd.DataFrame(combined_data, columns=['Map', 'Category', 'Value'])

# Set up the plot
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Create spaghetti plot
pt.half_violinplot(x='Category', y='Value', data=df, dodge=True, linewidth=1, palette=["lightblue", "lightgreen"], ax=ax)

# Add histograms on the sides
pt.stripplot(x='Category', y='Value', data=df, jitter=True, ax=ax, palette=["lightblue", "lightgreen"], dodge=True, linewidth=1, size=1.5)

# Customize plot
plt.title('Cortical vs Subcortical Correlation')
plt.xlabel('Category')
plt.ylabel('Correlation Value')

# Show plot
plt.tight_layout()
plt.show()




'''
##### Correlation Calculation for Mean Data (Group Level)

# Correlation coefficient for
corr_coeff_mean, p_value = pearsonr(corr_values_mean_cortical, corr_values_mean_subcortical)
print("Correlation coefficient for the mean cortical vs. whole-brain data:", corr_coeff_mean)
# -0.34626341820901974

##### Correlation Calculation for Single Subject Data (Individual Level)

annotation_sources = ['ding2010', 'hesse2017', 'kaller2017', 'alarkurtti2015', 'jaworska2020', 'sandiego2015',
                      'smith2017', 'sasaki2012', 'fazio2016', 'gallezot2010', 'radnakrishnan2018']

# Calculate the correlation coefficients
correlation_coefficients = {}
for source in annotation_sources:
    correlation_coefficients[source] = np.corrcoef(corr_values_ss_cortical_array[source], corr_values_ss_subcortical_array[source])[0, 1]

# Print the correlation coefficients
for key, value in correlation_coefficients.items():
    print(f"Correlation coefficient for {key}: {value}")

#Correlation coefficient for ding2010: 0.8888886255917567
#Correlation coefficient for hesse2017: 0.5393758133210639
#Correlation coefficient for kaller2017: 0.44510346978096765
#Correlation coefficient for alarkurtti2015: 0.10115170205649535
#Correlation coefficient for jaworska2020: 0.3986834663405925
#Correlation coefficient for sandiego2015: 0.38117053615017316
#Correlation coefficient for smith2017: 0.35904773674853996
#Correlation coefficient for sasaki2012: 0.4575705729532519
#Correlation coefficient for fazio2016: 0.535443141934806
#Correlation coefficient for gallezot2010: 0.8180894565459425
#Correlation coefficient for radnakrishnan2018: 0.4698982491796577
'''

