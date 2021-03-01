from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]











import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


bankdata = pd.read_csv(r"D:\Doctora99\Doctora-term1-9908\ML.DrManthouri\HW\SVM\bill_authentication.csv")
bankdata.shape
bankdata.head()

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=None)

###############################
logreg = LogisticRegression()
#logreg = LogisticRegression(penalty='none', solver='saga')
#logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
#logreg = LogisticRegression(penalty='l1', solver='liblinear')
#logreg = LogisticRegression(penalty='l2', solver='liblinear')

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print("############ Logistic Regression ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

###############################

# penalty is a string ('l2' by default) that decides whether there is regularization and which approach to use. Other options are 'l1', 'elasticnet', and 'none'.
#'liblinear' solver doesn’t work without regularization.
#'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
#'saga' is the only solver that supports elastic-net regularization.
#(penalty='none', solver='liblinear', random_state=0)

###############################