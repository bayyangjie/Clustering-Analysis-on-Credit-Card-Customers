## Project objectives
- Utilize Python to apply clustering techniques to group customers based on their credit card usage patterns to tailor marketing strategies and services effectively
- Provide insights and recommendations for the company based on the identified customer segments
- Discuss how the company can leverage the identified segments for targeted marketing or service improvements
  
## Learning points
- Handle missing values by filling them with median value
- Drop non-required variables
- Plot visualizations to understand the correlation between variables and distribution of values in the variables
- Elow plot creation to determine the optimal n clusters to be used in Clustering
- Plotted clusters visualizations to show centroid values of each cluster within individual features

<br>

### Data cleaning/processing

```
# Checking for the sum of missing values in each column
missing_values = df_ECA.isnull().sum()
print("\nCount of missing values in each column:")
print(missing_values)
```
Only two columns 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS' have missing values in them.

```
# Identify numeric columns
cols_missing_values = ['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']

# Fill missing values with the median in the specified columns
for col in cols_missing_values:
    median_value = df_ECA[col].median()
    df_ECA[col].fillna(median_value, inplace=True)
```
Converting the data types of those columns with missing values into numerical, then using the for loop to iterate through the columns and fill any NA values in each column with the respective 
median value.

```
# Dropping non-required columns
###### Drop non-numeric/non-required columns ######
df_ECA.drop('CUST_ID', axis=1, inplace=True)
```

```
# Scaling numerical features - Standardization
scaler = StandardScaler()
numerical_columns = df_ECA.select_dtypes(include=['float64', 'int64']).columns
numeric_df=df_ECA[numerical_columns]
```
Standardization is performed to ensure that all the variables in the dataset are on the same scale. This prevents a feature from dominating another due to the large difference in scale. 

<br>

## Visualisations

### Correlation heatmap: <br>
![correlation](https://github.com/bayyangjie/Foundation-to-Python-for-AI/assets/153354426/bc9b2ea7-fcec-4e6e-b156-858a96d519d7) <br> <br>

### Distribution of each numerical column: <br>
![distribution](https://github.com/bayyangjie/Foundation-to-Python-for-AI/assets/153354426/5634a94a-a788-47d0-8bfe-7d945e4efef1) <br> <br>

### Elbow plots:
```
# Elbow plot to find optimal n clusters for clustering step
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

The elbow plot shows that the optimal number of clusters is '4' as the point where the gradient of the slope decreases the most.

```
# Elbow method to determine n_components for PCA step
pca = PCA()
pca.fit(standardized_data)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Elbow plot to identify n_components')
plt.show()
```
The elbow plot shows that the ideal n_components to be used for the PCA is '2'. 

### PCA
```
pca = PCA(n_components=2)
pca_result = pca.fit_transform(standardized_data)
```
PCA is performed to reduce the dimensionality of a dataset while preserving most of its variance/information. By representing data in terms of its principal components (which are linear combinations of the original features), PCA allows for a lower-dimensional representation that captures the essential structure of the data. In this case, the output of the 2 columns summarize the variability in the data in a lower-dimensional space

### Clustering
```
# Perform clustering using PCA data
pca_result_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(2)])
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, max_iter=300, random_state=42)
labels = kmeans.fit_predict(pca_result)

unique_labels = sorted(set(labels))
legend_labels = [f'Cluster {label}' for label in unique_labels]

# Checking the quality of clusters using Silhouette score
silhouette_avg = silhouette_score(pca_result, labels)
print("Silhouette Score:", silhouette_avg)
```
The silhouette score is a measure of how similar a data point is to its own cluster compared to other clusters. The silhouette score ranges from -1 to 1 where a high value indicates that the data point is well matched to its own cluster and poorly matched to neighbouring clusters. In this case, the score returned a value of 0.40735770812035293 which is deemed to be a reasonably good clustering result.

```
# Plotting visualizations of the clusters
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
![Image 3](https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Images/Kmeans_clustering.png?raw=true) <br> <br>

The clustering output shows that there are 4 distinct clusters as stated by the number of n_clusters. 
Cluster 3 forms the majority as the cluster is the biggest in size and Cluster 0 forms the smallest size. The location of the cluster being at the bottom left corner is due to standardization that was being applied to the dataset. Centering of the clusters around the centre suggests a few things. Each cluster contains a comparable number of points, and the points are not overly concentrated in specific regions within the clusters. The centering of clusters might also indicate that the clustering algorithm has found an optimal configuration where clusters are well-separated and balanced.


### Obtaining centroid values
```
# Display the mean values (centroids) of each feature within each cluster
clustered_data = pd.DataFrame(data=standardized_data, columns=[f'Feature{i+1}' for i in range(standardized_data.shape[1])])
clustered_data['Cluster'] = labels
centroids = clustered_data.groupby('Cluster').mean()
print("Centroids (Mean Values) for Each Cluster:")
print(centroids)
```
The centroids obtained from k-means clustering represent the average values of each feature within each cluster as shown in the table below. The centroids can provide insights into the impact of each variable within each cluster. A larger magnitude indicates a stronger impact of that feature within the cluster. 

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

Bar plot creation to show centroid values per cluster of each feature:
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

The bar plot was created firstly by transposing the dataframe ‘centroids’ to have features as rows and clusters as columns. The plot shows the centroid magnitude of each variable that is being sub-categorized into the clusters. From that, we can determine the impact of each variable within each cluster and derive insights from it.
