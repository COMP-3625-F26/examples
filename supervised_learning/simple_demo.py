from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import itertools


# load the Iris dataset and print it
df = load_iris(as_frame=True)['frame']
print(df)

# reduce to just 2 dimensions and 2 kinds of flowers (keep the problem simple for illustration)
features_to_keep = ['sepal length (cm)', 'petal length (cm)']
classes_to_keep = [1, 2]
X = df.loc[df['target'].isin(classes_to_keep), features_to_keep].to_numpy()
y = df.loc[df['target'].isin(classes_to_keep), 'target'].to_numpy()


# plot the data
fig, ax = plt.subplots(1, 2)
print(classes_to_keep[0], type(classes_to_keep[0]))
ax[0].scatter(X[:,0], X[:,1], c=y)
ax[0].set_xlabel(features_to_keep[0])
ax[0].set_ylabel(features_to_keep[1])


# fit a decision tree model to the data
model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
model.fit(X=X, y=y)

# plot the learned decision tree
if isinstance(model, DecisionTreeClassifier):
    plt.figure()
    plot_tree(model, fontsize=10, feature_names=features_to_keep)

# visualize how the model divides up the space into two classes
# by using it to classify test points that span the space
x_test_pts = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
y_test_pts = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
test_coords = np.array(list(itertools.product(x_test_pts, y_test_pts)))
if hasattr(model, 'predict_proba'):
    classifications = [model.predict_proba([test_coord])[0][1] for test_coord in test_coords]
else:
    classifications = [model.predict([test_coord])[0] for test_coord in test_coords]
ax[1].set_xlabel(features_to_keep[0])
ax[1].set_ylabel(features_to_keep[1])
scatterplt = ax[1].scatter(test_coords[:,0], test_coords[:, 1], c=classifications)
if hasattr(model, 'predict_proba'):
    cbar = fig.colorbar(scatterplt, ax=ax[1])
    cbar.set_label(f'probability of belonging to class {classes_to_keep[1]}')

plt.show()