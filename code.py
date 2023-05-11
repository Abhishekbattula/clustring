#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
import itertools as iter


# In[2]:


# Loading the forest area data
forest_area_data = pd.read_csv("forest_area.csv", skiprows=4)
forest_area_data = forest_area_data[["Country Name", "2019"]]
forest_area_data = forest_area_data.rename(columns={"Country Name": "Country", "2019": "Forest area"})
forest_area_data.head()


# In[3]:


# Loading the renewable energy consumption data
renewable_energy_data = pd.read_csv("renewable_energy.csv", skiprows=4)
renewable_energy_data = renewable_energy_data[["Country Name", "2019"]]
renewable_energy_data = renewable_energy_data.rename(columns={"Country Name": "Country", "2019": "Renewable Energy Consumption"})
renewable_energy_data.head()


# In[4]:


# Merge the two dataframes on the "Country" column
df = pd.merge(forest_area_data, renewable_energy_data, on="Country")
df.head()


# In[5]:


#checking missing values
df.isnull().sum()


# In[6]:


# Remove rows with missing values
df.dropna(inplace=True)


# In[7]:


#checking missing values after dropping them
df.isnull().sum()


# In[8]:


# Select the columns for normalization
columns_to_normalize = ['Forest area', 'Renewable Energy Consumption']

# Create a scaler object
scaler = StandardScaler()

# Normalize the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# In[9]:


df[columns_to_normalize]


# In[10]:


#Applying Kmeans Clustering
# Select the columns for clustering
columns_for_clustering = ['Forest area', 'Renewable Energy Consumption']

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[columns_for_clustering])
df['Cluster']


# In[11]:


#Plot the results

# Set the figure size
plt.figure(figsize=(10, 8))

# Scatter plot of the clusters
plt.scatter(df['Forest area'], df['Renewable Energy Consumption'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Forest Area (Normalized)')
plt.ylabel('Renewable Energy Consumption (Normalized)')
plt.title('Clustering Results')
plt.colorbar(label='Cluster')
plt.show()


# The code above generates a scatter plot that visualizes the clustering results based on the normalized 'Forest area' and 'Renewable Energy Consumption' variables. Each data point is assigned a color corresponding to its cluster membership, allowing for visual identification of different clusters.
# 
# This line adds a color bar legend on the side of the plot, indicating the mapping between cluster colors and cluster labels. The label parameter sets the title of the color bar to 'Cluster'.
# 
# The code creates a scatter plot where the x-axis represents the normalized values of the 'Forest area' variable and the y-axis represents the normalized values of the 'Renewable Energy Consumption' variable. Each data point is assigned a color based on its cluster membership, which is specified using the 'Cluster' column of the 'merged_data' DataFrame. The cmap='viridis' parameter sets the color map to 'viridis' for better visualization.

# In[12]:


# Extract the relevant columns from the dataset
data = pd.concat([forest_area_data['Forest area'], renewable_energy_data['Renewable Energy Consumption']], axis=1)
data.columns = ['Forest Area', 'Renewable Energy Consumption']

# Drop any rows with missing values
data.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Perform clustering using K-means
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the dataframe
data['Cluster'] = labels

# Plot cluster membership and cluster centers
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(data['Forest Area'], data['Renewable Energy Consumption'], c=labels, cmap='viridis', alpha=0.8)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

plt.xlabel('Forest Area (sq. km)')
plt.ylabel('Renewable Energy Consumption (percentage of total final energy consumption)')
plt.title('Clustering Results of Forest Area vs Renewable Energy Consumption')
plt.legend()
plt.show()


# By examining the plot and analyzing the clustering results, below are the findings.
# 
# * The clustering results reveal distinct patterns in the relationship between forest area and renewable energy consumption.
# * The three different clusters represent different combinations of forest area and renewable energy consumption.
# * Cluster analysis can help identify countries or regions with similar characteristics and potentially provide insights into their environmental and energy policies.
# * Further analysis and interpretation can be done by examining the cluster characteristics, such as average forest area and renewable energy consumption, and comparing them to other socio-economic or environmental indicators.
# * This analysis can be used to inform decision-making processes, such as targeting policies or investments towards countries with low forest area and high renewable energy consumption to promote sustainable development and environmental conservation.

# In[13]:


# Define the function for the model you want to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Extract the relevant columns from the dataset
forest_area = data['Forest Area'].astype(float)
renewable_energy = data['Renewable Energy Consumption'].astype(float)

# Define the function for estimating confidence ranges
def err_ranges(x, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf(alpha / 2))
    lower = polynomial_function(x, *(popt - z_score * perr))
    upper = polynomial_function(x, *(popt + z_score * perr))
    return lower, upper

# Fit the model to the data
popt, pcov = curve_fit(polynomial_function, forest_area, renewable_energy)

# Make predictions for future values
forest_area_future = np.linspace(np.min(forest_area), np.max(forest_area), 100)
renewable_energy_pred = polynomial_function(forest_area_future, *popt)

# Calculate confidence ranges
lower, upper = err_ranges(forest_area_future, popt, pcov)

# Plot the data, best fitting function, and confidence range
plt.figure(figsize=(10, 6))
plt.scatter(forest_area, renewable_energy, label='Data')
plt.plot(forest_area_future, renewable_energy_pred, color='red', label='Best Fit')
plt.fill_between(forest_area_future, lower, upper, color='gray', alpha=0.3, label='Confidence Range')
plt.xlabel('Forest Area (sq. km)')
plt.ylabel('Renewable Energy Consumption (percentage of total final energy consumption)')
plt.title('Polynomial Fit of Forest Area vs Renewable Energy Consumption')
plt.legend()
plt.show()


# The analysis aims to explore the relationship between forest area and renewable energy consumption. By fitting a polynomial function to the dataset, we can observe the trend and make predictions for future values.
# 
# The scatter plot showcases the original data points, with each point representing a country's forest area and corresponding renewable energy consumption. The best fitting polynomial curve is depicted in red, indicating the trend captured by the model.
# 
# The curve suggests a non-linear relationship between forest area and renewable energy consumption. As the forest area increases, the consumption of renewable energy tends to rise, but the relationship is not strictly linear. Instead, it exhibits a quadratic pattern, indicating a possible saturation point where additional forest area may have diminishing returns in terms of renewable energy consumption.
# 
# The shaded area represents the confidence range of the best fit curve. It provides an estimation of the possible range of values for renewable energy consumption corresponding to different forest area values. The wider the shaded area, the greater the uncertainty in the predictions.
# 
# With this analysis, we can make predictions for future values of renewable energy consumption based on the forest area. However, it's important to note that the predictions are subject to the assumptions of the polynomial model and the inherent uncertainty in the data.

# In[14]:


for l, u in zip(lower, upper):
    print(f"Lower limit: {l:.2f}, Upper limit: {u:.2f}")


# This modified code will iterate over the arrays lower and upper, which represent the lower and upper confidence limits, respectively, for the predicted values of Renewable Energy Consumption. It will print the lower and upper limits for each corresponding prediction.
# 
# For each iteration, the code will print the lower limit for Renewable Energy Consumption with two decimal places, followed by the upper limit with two decimal places. These limits represent the range within which the predicted values of Renewable Energy Consumption are expected to fall with a certain level of confidence. The lower limit indicates the minimum expected value, while the upper limit indicates the maximum expected value.
# 
# Printing these limits provides additional information about the uncertainty associated with the predictions made by the polynomial model for Renewable Energy Consumption. It allows us to understand the range of potential values for Renewable Energy Consumption and provides insights into the level of confidence we can have in the predictions.

# In[15]:


# Define the function for the model you want to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c


# Extract the relevant columns from the dataset
forest_area = data['Forest Area'].astype(float)
renewable_energy = data['Renewable Energy Consumption'].astype(float)


# Define the function for estimating confidence ranges
def err_ranges(x, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf(alpha / 2))
    lower = polynomial_function(x, *(popt - z_score * perr))
    upper = polynomial_function(x, *(popt + z_score * perr))
    return lower, upper


# Perform clustering on the dataset
X = np.column_stack((forest_area, renewable_energy))
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)

# Add cluster labels to the dataframe
data['Cluster'] = labels

# Initialize a list to store the fitting results for each cluster
fit_results = []

# Iterate over each cluster
for cluster_id in range(3):
    # Get the data points belonging to the current cluster
    cluster_data = data[data['Cluster'] == cluster_id]
    
    # Check if the cluster has sufficient data points for fitting a curve with three parameters
    if len(cluster_data) < 3:
        print(f"Cluster {cluster_id} does not have enough data points for fitting a curve.")
        continue
    
    # Fit the model to the data
    popt, pcov = curve_fit(polynomial_function, cluster_data['Forest Area'], cluster_data['Renewable Energy Consumption'])
    
    # Store the fitting results
    fit_results.append((popt, pcov))

# Plot the data, best fitting function, and confidence range for each cluster
plt.figure(figsize=(10, 6))
for cluster_id, fit_result in enumerate(fit_results):
    popt, pcov = fit_result
    cluster_data = data[data['Cluster'] == cluster_id]
    forest_area_future = np.linspace(np.min(forest_area), np.max(forest_area), 100)
    renewable_energy_pred = polynomial_function(forest_area_future, *popt)
    lower, upper = err_ranges(forest_area_future, popt, pcov)
    plt.scatter(cluster_data['Forest Area'], cluster_data['Renewable Energy Consumption'], label=f'Cluster {cluster_id}')
    plt.plot(forest_area_future, renewable_energy_pred, label=f'Cluster {cluster_id} - Best Fit')
    plt.fill_between(forest_area_future, lower, upper, color='gray', alpha=0.3, label=f'Cluster {cluster_id} - Confidence Range')

plt.xlabel('Forest Area (sq. km)')
plt.ylabel('Renewable Energy Consumption (percentage of total final energy consumption)')
plt.title('Polynomial Fit by Cluster')
plt.legend()
plt.show()


# The chart displays the polynomial fit and confidence ranges for each cluster in the dataset, where the variables are 'Forest Area' and 'Renewable Energy Consumption'.
# 
# Each cluster is represented by a distinct color, and the data points belonging to each cluster are plotted on the chart. The scatter points show the relationship between forest area and renewable energy consumption for each cluster.
# 
# For each cluster, a second-degree polynomial function is fitted to the data using the curve_fit function. The resulting best-fit curve is plotted for each cluster, indicating the estimated relationship between forest area and renewable energy consumption within that cluster.
# 
# Additionally, confidence ranges are calculated for each cluster's fitted curve using the err_ranges function. The confidence ranges represent the range of values within which the true relationship between forest area and renewable energy consumption is likely to fall. The confidence range is visualized as a shaded region around the fitted curve, with a lighter shade indicating higher confidence.
# 
# The chart allows for a visual comparison of the fitted curves and their associated confidence ranges across different clusters. It provides insights into the variations in the relationship between forest area and renewable energy consumption within the dataset. Clusters with similar trends in forest area and renewable energy consumption are expected to have fitted curves that follow a similar pattern, while clusters with distinct trends may exhibit different shapes in their fitted curves.
# 
# By analyzing the chart, one can identify clusters with similar patterns and observe the overall trends in the relationship between forest area and renewable energy consumption. It can help in understanding the diversity and patterns in the dataset and support decision-making related to forest conservation and renewable energy planning.
