# WINSCONSIN CANCER DATASET

### About dataset
- This dataset is available at [UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
- Missing value: No
- Number of samples: 569
- Number of feature: 32

### Feature information
- (1) ID number
- (2) Label: M (malignant), B(benign)
- (3-32) Ten real value features are computed for each cell nucleus
  - a) radius (mean of distances from center to points on the perimeter) 
  - b) texture (standard deviation of gray-scale values) 
  - c) perimeter 
  - d) area 
  - e) smoothness (local variation in radius lengths) 
  - f) compactness (perimeter^2 / area - 1.0) 
  - g) concavity (severity of concave portions of the contour) 
  - h) concave points (number of concave portions of the contour) 
  - i) symmetry 
  - j) fractal dimension ("coastline approximation" - 1)
  
  
### Target
- main.py
  - Using linear model and logistic classification for training and inference
- main_tf.py:
  - Using deep learning feed forward network with 1,2,3,4,5 hidden layers
  - Implement with tensorflow
  - Chossing the best model