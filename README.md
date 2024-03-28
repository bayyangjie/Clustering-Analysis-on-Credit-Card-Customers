## Project objectives
- Utilize Python for applying clustering techniques to group customers based on their credit card usage patterns to tailor marketing strategies and services effectively

## What was done
- Handle missing values by filling them with median value
- Drop non-required variables
- Plot visualizations to understand the correlation between variables and distribution of values in the variables
- Elow plot creation to determine the optimal n clusters to be used in Clustering
- Plotted clusters visualizations to show centroid values of each cluster within individual features

Handling missing values:
```
missing_values = df_ECA.isnull().sum()
print("\nCount of missing values in each column:")
print(missing_values)
```

Filling in missing values with the median:
```
for col in cols_missing_values:
    median_value = df_ECA[col].median()
    df_ECA[col].fillna(median_value, inplace=True)
```

Dropping non-required columns:
```
###### Drop non-numeric/non-required columns ######
df_ECA.drop('CUST_ID', axis=1, inplace=True)
```

Scaling numerical features:
```
scaler = StandardScaler()
numerical_columns = df_ECA.select_dtypes(include=['float64', 'int64']).columns
numeric_df=df_ECA[numerical_columns]
```

Performing PCA:
```
pca = PCA(n_components=2)
pca_result = pca.fit_transform(standardized_data)
```

Perform clustering using PCA data:
```
pca_result_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(2)])
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, max_iter=300, random_state=42)
labels = kmeans.fit_predict(pca_result)

unique_labels = sorted(set(labels))
legend_labels = [f'Cluster {label}' for label in unique_labels]
```

Checking the quality of clusters using Silhouette score:
```
silhouette_avg = silhouette_score(pca_result, labels)
print("Silhouette Score:", silhouette_avg)
```

Calculating centroid values of each feature within each cluster:
```
clustered_data = pd.DataFrame(data=standardized_data, columns=[f'Feature{i+1}' for i in range(standardized_data.shape[1])])
clustered_data['Cluster'] = labels
centroids = clustered_data.groupby('Cluster').mean()
print("Centroids (Mean Values) for Each Cluster:")
print(centroids)
```
Converting centroid data into a barplot:
```
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
```

## Visualizations

Correlation Heatmap:
```
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
```
![Image 1](https://github.com/bayyangjie/Data-Visualization-and-Storytelling/blob/main/Images/Picture%204.png?raw=true)) <br> <br>


Elbow plot to find optimal n clusters:
```
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
```
![Image 2](https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Images/elbow_plot_2.png?raw=true) <br> <br>


K Means Clustering:
```
pca_result_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(2)])
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, max_iter=300, random_state=42)
labels = kmeans.fit_predict(pca_result)

unique_labels = sorted(set(labels))
legend_labels = [f'Cluster {label}' for label in unique_labels]

# Checking the clustering quality using silhouette score
silhouette_avg = silhouette_score(pca_result, labels)
print("Silhouette Score:", silhouette_avg)

# Creating the clusters visualization
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
```
![Image 3](https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Images/elbow_plot_2.png?raw=true) <br> <br>


Bar plot showing centroid values per cluster of each feature:
```
# Display the mean values (centroids) of each feature within each cluster
clustered_data = pd.DataFrame(data=standardized_data, columns=[f'Feature{i+1}' for i in range(standardized_data.shape[1])])
clustered_data['Cluster'] = labels
centroids = clustered_data.groupby('Cluster').mean()
print("Centroids (Mean Values) for Each Cluster:")
print(centroids)

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

# Add labels and legends
plt.xlabel('Feature',fontsize=30)
plt.ylabel('Centroid Value (Magnitude)',fontsize=30)
plt.title('Centroid Values (Magnitude) for Different Features Across Clusters',fontsize=30)
plt.legend(title='Cluster', loc='upper right',fontsize=20)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.tight_layout()
plt.show()
```
![Image 4](https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Images/feature_cluster_barplot.png?raw=true)


