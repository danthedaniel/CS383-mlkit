CS383 - Machine Learning
===

Combination of machine learning operators implemented for Drexel's CS383 course.

The algorithms implemented here use the most naive implementations possible, so
they may be a good place to look when learning a new algorithm.

### Usage

The machine learning models conform to the sklearn API. They all have `fit` and
`transform`/`predict` methods as well as a `score` method. These are documented
in [base.py](mlkit/base.py) with the abstract classes `Estimator` and
`Transformer`.

### Example

```python
from mlkit import KNearest
from mlkit.util import train_test_split, accuracy
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, y_train, X_test, y_test = train_test_split(X, y, 0.75)
knn = KNearest(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Testing Accuracy: {}'.format(accuracy(y_pred, y_test)))
```
