# Clustering Analysis on Credit Card Customers

 Python code file: https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Project.py

## About the project
This project utilizes Python for applying clustering techniques to group customers based on their credit card usage patterns to tailor marketing strategies and services effectively. The dataset contains the usage behavior of around 9000 credit card holders during the last 6 months, including the following variables.

* In this project, K-Means clustering was the main method employed for the clustering analysis because the dataset consists of numerical attributes.
* Various steps were also performed prior to performing the clustering. For example, PCA was performed to reduce the dimensionality of the original feature space.
* The elbow plot was also created to identify the optimum "n_components" and number of clusters (n_clusters).

## About the dataset
The column variables in the dataset describe the credit card usage behaviour of around 9000 credit card holders over a period of 6 months and the individual customers represent the rows. The different column variables represent the various metrics for measuring the usage behaviour of a customer. The customer ID column was also dropped in this case because it is not required as part of the clustering analysis. 

## Visualisations

### Correlation Heatmap
Based on the correlation heatmap, there are mostly light colours depicting the relationship between variables. This suggests that the variables are not strongly correlated with each other. This means there is low correlation between the variables suggesting lower multicollinearity. Having low multicollinearity can potentially lead to a more reliable and interpretable modelling results.

![image](https://github.com/user-attachments/assets/b2c5c71c-7bfe-4372-90a1-8b588bf0fca1)

### Elbow Plot (n_components)
The n_components value ‘2’ was derived from the elbow plot method as shown below and using the standardized dataframe ‘standardized_data’ as the input.

![image](https://github.com/user-attachments/assets/2902aff7-50c8-48e8-9c61-16579b8c6b7d)

### Elbow Plot (optimal clusters)
With the standardization having performed earlier during the feature scaling step, the elbow method can be performed to identify the ideal n_clusters value. This was done using the standardized dataframe ‘standardized_data’. From the elbow plot below, we can infer that the point n=4 is the ideal number of clusters.

![image](https://github.com/user-attachments/assets/401b38d6-e006-4def-bdd2-750a352985e3)

### Clusters
The cluster plot shows result of the k-means clustering. Customers in Cluster 3 form the majority as it is the biggest cluster while Cluster 0 is the smallest. Cluster 0 and Cluster 1 are rather compact indicating that the customers of those clusters likely possess similar characteristics. On the other hand, larger, spread-out clusters like Cluster 2 and 3 might have more variability among the types of customers in each of the two clusters.  

![image](https://github.com/user-attachments/assets/542bfff8-04ef-462e-8395-57c9f53805c0)

### Barplot break down of clusters 
* 

![image](https://github.com/user-attachments/assets/36b48e4c-5f38-4422-9db6-439476f1d062)

## Limitations and Improvements
* Improvements could be made to the dataset to include other attributes such as demographic information of customers such as age, gender, ethnicity, marital status, employment status and so on. With the inclusion of such details, it could benefit the company greatly by designing more accurate incentive/reward programs and advertisements to be targeted towards more specific groups of customers. 

* Besides KMeans clustering, DBSCAN is also another technique that could be used for clustering. DBSCAN is useful for working with datasets that have complex structures and it is also insensitive to scaling which means it can also handle datasets with different scales without the need for normalization or standardization.
