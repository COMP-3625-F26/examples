from sklearn.datasets import make_classification, make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# create a toy dataset for clustering
X, y = make_blobs(n_samples=100, n_features=2, centers=6)

inertias = []
for n in range(1, 11):
    # instantiate k-means clustering algorithm
    kmeans = KMeans(n_clusters=n)
    # run the clustering
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# run a k-means clustering with k=5
# then use the .predict method to get cluster assignments for each instance in the dataset
# plot these assignments
# YOUR CODE HERE


# plot
fig, ax = plt.subplots(1, 2)
ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set_title('true cluster memberships')
ax[1].plot(inertias)

plt.show()