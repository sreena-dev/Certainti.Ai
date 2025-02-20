
# Scikit-Learn 1.5.2 

## What is Scikit-Learn?
**Scikit-learn** is a popular **machine learning (ML) library** in Python. It provides efficient tools for:
- **Supervised Learning** (classification, regression)
- **Unsupervised Learning** (clustering, dimensionality reduction)
- **Model Selection & Evaluation** (cross-validation, hyperparameter tuning)
- **Preprocessing & Feature Engineering**
- **Pipelines & Automation**

---

## Installation
To install the latest stable version:
```sh
pip install scikit-learn
```
If `1.5.2` is available, install it using:
```sh
pip install scikit-learn==1.5.2
```

---

## Key Features & Modules

### 1. Supervised Learning
**Classification Algorithms**
- `LogisticRegression`
- `SVC` (Support Vector Machines)
- `RandomForestClassifier`
- `KNeighborsClassifier`
- `MLPClassifier` (Neural Networks)

**Regression Algorithms**
- `LinearRegression`
- `Ridge`, `Lasso` (Regularization)
- `SVR` (Support Vector Regression)
- `RandomForestRegressor`
- `MLPRegressor` (Neural Network)

**Example: Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```

---

### 2. Unsupervised Learning
**Clustering**
- `KMeans`
- `DBSCAN`
- `AgglomerativeClustering`

**Dimensionality Reduction**
- `PCA` (Principal Component Analysis)
- `TSNE` (t-SNE for visualization)

**Example: K-Means Clustering**
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
print("Cluster Centers:", kmeans.cluster_centers_)
```

---

### 3. Model Selection & Evaluation
**Cross-Validation**
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
```

**Hyperparameter Tuning**
- `GridSearchCV`
- `RandomizedSearchCV`

---

### 4. Preprocessing & Feature Engineering
**Feature Scaling**
- `StandardScaler`
- `MinMaxScaler`

**Feature Selection**
- `SelectKBest`
- `VarianceThreshold`

**Example: Standardizing Data**
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```

---

### 5. Pipelines & Automation
**Pipeline Example**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear"))
])

pipeline.fit(X_train, y_train)
print("Pipeline accuracy:", pipeline.score(X_test, y_test))
```

---

## What's New in Latest Versions?
Each new version brings **performance improvements, bug fixes, and new features**.  
To check for the latest version:
```sh
pip install -U scikit-learn
```
Or visit the [Scikit-Learn Release Notes](https://scikit-learn.org/stable/whats_new.html).

---

## Final Thoughts
- **Scikit-learn is a go-to library for ML**  
- **Supports both beginners and advanced users**  
- **Great for building ML models quickly**  
