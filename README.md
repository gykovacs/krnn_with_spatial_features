# KRNN with spatial features

Implementation of the technique described in
        
_László et al (2018): Improving the Performance of the k Rare Class Nearest Neighbor Classifier by the Ranking of Point Patterns. In: Lecture Notes in Computer Science, vol 10833, p. 265-283_

* doi: [https://doi.org/10.1007/978-3-319-90050-6_15](https://doi.org/10.1007/978-3-319-90050-6_15)
* link: [https://link.springer.com/chapter/10.1007/978-3-319-90050-6_15](https://link.springer.com/chapter/10.1007/978-3-319-90050-6_15)
* preprint: [PREPRINT](https://drive.google.com/open?id=1aQct5L6DgYvRnE6wDMKzekhZvxEyktGz)

## Installation

```bash
pip install git+https://github.com/gykovacs/krnn_with_spatial_features
```

## Sample usage

```python
# importing the dataset package from sklearn
import sklearn.datasets as sd

# import the KRNN_SF classifier
from KRNN_SF import KRNN_SF

# loading the IRIS dataset
X, y= sd.load_iris(return_X_y= True)

# turning the IRIS multi-classification problem into an unbalanced binary classification
y[y == 2]= 1

# fitting and predicting
krnn_sf= KRNN_SF(correction= 'r1')
krnn_sf.fit(X, y)
krnn_sf.predict_proba(X)
```
