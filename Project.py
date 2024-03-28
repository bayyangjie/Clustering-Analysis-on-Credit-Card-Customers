#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:42:59 2024

@author: ASUS
"""

import os
import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



####### Read in dataset ######
df_ECA = pd.read_csv('/Users/ASUS/Library/CloudStorage/OneDrive-Personal/Desktop/SUSS/AIB503/ECA/AIB503_ECA_Data.csv',header=None, skiprows=1, sep=',')
df_ECA = df_ECA.reset_index(drop=True)
df_ECA.columns = df_ECA.iloc[0]
df_ECA = df_ECA.drop(0)
print(df_ECA)

#####  Data pre-processing  ######

# Handling missing values

## Checking for the sum of missing values in each column
missing_values = df_ECA.isnull().sum()
print("\nCount of missing values in each column:")
print(missing_values)

# Identify numeric columns
cols_missing_values = ['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']

# Convert the specified columns to numeric, replacing non-numeric values with NaN
df_ECA[cols_missing_values] = df_ECA[cols_missing_values].apply(pd.to_numeric, errors='coerce')

# Fill missing values with the median
for col in cols_missing_values:
    median_value = df_ECA[col].median()
    df_ECA[col].fillna(median_value, inplace=True)

# Verify that there are no more missing values in all columns
missing_values_after = df_ECA.isnull().sum()
print("\nCount of missing values in each column after replacement:")
print(missing_values_after)

###### Drop non-numeric columns ######
df_ECA.drop('CUST_ID', axis=1, inplace=True)

# Print the DataFrame after dropping the column
print(df_ECA)


# Converting all column data types to float
df_ECA = df_ECA.astype(float)



###### Scaling numerical features - Standardization ######
scaler = StandardScaler()
numerical_columns = df_ECA.select_dtypes(include=['float64', 'int64']).columns
numeric_df=df_ECA[numerical_columns]

standardized_data=scaler.fit_transform(numeric_df)# to be used for PCA 


####### Visualisation #1 - Correlation Heatmap ######

# Calculate the correlation heatmap
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

###### Visualization #2 - Distribution of each numerical column ######
numeric_df.hist(bins=10, edgecolor='black', figsize=(20,10), grid=False)
plt.tight_layout()
plt.show()

####### Visualisation #3 - Histogram pairplots ######

# Plot pairplot of histograms
selected_cols=numeric_df.iloc[:,[0,2,3,5,6,7,8,12,13,14,15,16]] # selected relevant columns
plt.figure(figsize=(40,10))
sns.pairplot(selected_cols)
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.show()


###### Elbow method to determine n_components to be used in PCA() ######
pca = PCA()
pca.fit(standardized_data)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Elbow plot to identify n_components')
plt.show()


###### Elbow method to determine n_clusters - Using original features ######
inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=20, max_iter=300, random_state=111)
    kmeans.fit(standardized_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(25,20))
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.title('Elbow Plot',fontsize=30)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.xlabel('Number of Clusters',fontsize=24)
plt.ylabel('Inertia',fontsize=24)
plt.show()

###### Performing PCA ######
pca = PCA(n_components=2)
pca_result = pca.fit_transform(standardized_data)


###### Perform Clustering using PCA data ######
pca_result_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(2)])
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, max_iter=300, random_state=42)
labels = kmeans.fit_predict(pca_result)

unique_labels = sorted(set(labels))
legend_labels = [f'Cluster {label}' for label in unique_labels]

# Checking the clustering quality using silhouette score
silhouette_avg = silhouette_score(pca_result, labels)
print("Silhouette Score:", silhouette_avg)

####### Creating the clusters visualisation ######
plt.figure(figsize=(8, 6))
for cluster_label in unique_labels:
    cluster_indices = labels == cluster_label
    plt.scatter(pca_result_df.loc[cluster_indices, 'PC1'], pca_result_df.loc[cluster_indices, 'PC2'], label=f'Cluster {cluster_label}', s=50)

plt.title('K-Means Clustering on PCA Data',fontsize=15)
plt.xlabel('Principal Component 1',fontsize=15)
plt.ylabel('Principal Component 2',fontsize=15)
plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Display the mean values (centroids) of each feature within each cluster
clustered_data = pd.DataFrame(data=standardized_data, columns=[f'Feature{i+1}' for i in range(standardized_data.shape[1])])
clustered_data['Cluster'] = labels
centroids = clustered_data.groupby('Cluster').mean()
print("Centroids (Mean Values) for Each Cluster:")
print(centroids)


###### Converting the centroid data into bar chart #######
# Take the absolute values of the centroids DataFrame
centroids_abs = centroids.abs()

# Transpose the DataFrame to have features as rows and clusters as columns
centroids_transposed = centroids_abs.T

# Set the figure size to make the plot larger
fig, ax = plt.subplots(figsize=(22, 10))

# Increase the width of the bars using the width parameter
bar_width = 0.6   
gap_width = 2
centroids_transposed.plot(kind='bar', stacked=False, ax=ax, width=bar_width)


# Add labels and legend
plt.xlabel('Feature',fontsize=30)
plt.ylabel('Centroid Value (Magnitude)',fontsize=30)
plt.title('Centroid Values (Magnitude) for Different Features Across Clusters',fontsize=30)
plt.legend(title='Cluster', loc='upper right',fontsize=20)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.tight_layout()
plt.show()


