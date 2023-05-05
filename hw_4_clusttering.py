import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("display.max_columns", None)  # Show all columns
# Prevent wrapping to next line
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.width", None)  # Auto-detect the display width

# pd.set_option("display.width", <your_desired_width>)  # Set a specific width if needed
df = pd.read_excel('/Users/surabhisuman/Downloads/assignment.xlsx')
print(df.keys())

df = df.drop(['University name'], axis=1)

scaler = MinMaxScaler()  # initialize
scaler.fit(df)
scaled_df = scaler.transform(df)

wcv = []
silk_score = []

for i in range(2, 15):
    km = KMeans(n_clusters=i, random_state=0)  # initialize
    km.fit(scaled_df)  # train: #identift clusters
    # wcv and silk score
    wcv.append(km.inertia_)  # wcv
    (silk_score.append(silhouette_score(scaled_df, km.labels_)))
    print("For nclusters = {}, silhouette score is {})".format(
        i, silhouette_score(scaled_df, km.labels_)))

# It is seen that the silhouette score is the highest for cluster 2 which is 0.2618

# plotting the wcv
plt.plot(range(2, 15), wcv)
plt.xlabel('No of Clusters')
plt.ylabel('Within Cluster Variation')
plt.show()

# Plotting the silk score
plt.plot(range(2, 15), silk_score)
plt.xlabel('No of Clusters')
plt.ylabel('Silk Score')
plt.show()

# so we go with 2 clusters
km3 = KMeans(n_clusters=2, random_state=0)  # initialize
km3.fit(scaled_df)  # train: identify clusters

df['labels'] = km3.labels_


# interpreting the results
df.groupby('labels').mean()


# Cluster 0
df1 = df.loc[df['labels'] == 0].describe()
df1.to_csv('/Users/surabhisuman/Downloads/cluster0.csv')
print(df1)


# Cluster 1
df2 = df.loc[df['labels'] == 1].describe()
df2.to_csv('/Users/surabhisuman/Downloads/cluster1.csv')
print(df.loc[df['labels'] == 1].describe())


# Dendogram**************************************************

# Using the dendrogram to find the optimal number of clusters
linked = linkage(scaled_df, 'ward')  # gets a n-1 *4 matrix
dendrogram(linked)  # uses the matrix to get to draw the dendrogram
plt.title("Dendrogram")
plt.xlabel('University Name')
plt.ylabel('Euclidean Distance')
plt.show()

threshold = 5
clusters = fcluster(linked, threshold, criterion='distance')
print("Cluster labels: ", clusters)

# We will go with 4 clusters
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
hc.fit(scaled_df)

# Adding the labels to df
df['labels'] = hc.labels_

# interpret clusters
print(df.groupby('labels').mean())
df3 = df.groupby('labels').mean()
df3.to_csv('/Users/surabhisuman/Downloads/groupbymean.csv')

# Cluster 0
print(df.loc[df['labels'] == 0].describe())
df4 = df.loc[df['labels'] == 0].describe()
df4.to_csv('/Users/surabhisuman/Downloads/dendcl0.csv')

# Cluster 1
print(df.loc[df['labels'] == 1].describe())
df5 = df.loc[df['labels'] == 1].describe()
df5.to_csv('/Users/surabhisuman/Downloads/dendcl1.csv')

# Cluster 2
print(df.loc[df['labels'] == 2].describe())
df6 = df.loc[df['labels'] == 2].describe()
df6.to_csv('/Users/surabhisuman/Downloads/dendcl2.csv')

# Cluster 3
print(df.loc[df['labels'] == 3].describe())
df7 = df.loc[df['labels'] == 3].describe()
df7.to_csv('/Users/surabhisuman/Downloads/dendcl3.csv')
