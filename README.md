# Clustering Analysis on Credit Card Customers

To run this project, use the following code file: <br> <br>
[View the full code here] (https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Project.py)

This project utilizes Python for applying clustering techniques to group customers based on their credit card usage patterns to tailor marketing strategies and services effectively. The dataset contains the usage behavior of around 9000 credit card holders during the last 6 months, including the following variables.

* In this project, K-Means clustering was the main method employed for the clustering analysis because the dataset consists of numerical attributes.
* Various steps were also performed prior to performing the clustering. For example, PCA was performed to reduce the dimensionality of the original feature space.
* The elbow plot was also created to identify the optimum "n_components" and number of clusters (n_clusters).

# Visualisations

### Correlation Heatmap

![image](https://github.com/user-attachments/assets/b2c5c71c-7bfe-4372-90a1-8b588bf0fca1)

### Elbow Plot (n_components)

![image](https://github.com/user-attachments/assets/2902aff7-50c8-48e8-9c61-16579b8c6b7d)

### Elbow Plot (optimal clusters)

![image](https://github.com/user-attachments/assets/401b38d6-e006-4def-bdd2-750a352985e3)

### Clusters

![image](https://github.com/user-attachments/assets/542bfff8-04ef-462e-8395-57c9f53805c0)

### Clusters represented as barplot visuals 

![image](https://github.com/user-attachments/assets/36b48e4c-5f38-4422-9db6-439476f1d062)

# Limitations and Improvements
* Improvements could be made to the dataset to include other attributes such as demographic information of customers such as age, gender, ethnicity, marital status, employment status and so on. With the inclusion of such details, it could benefit the company greatly by designing more accurate incentive/reward programs and advertisements to be targeted towards more specific groups of customers. 

* Besides KMeans clustering, DBSCAN is also another technique that could be used for clustering. DBSCAN is useful for working with datasets that have complex structures and it is also insensitive to scaling which means it can also handle datasets with different scales without the need for normalization or standardization.

### [View the full code here] (https://github.com/bayyangjie/Foundation-to-Python-for-AI/blob/main/Project.py)
