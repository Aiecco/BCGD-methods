# Gradient Methods for Multi-class Logistic Regression

This project explores the application and performance of various gradient-based methods to optimize a multi-class logistic regression model. We compare the performance of:

- Full Gradient Descent
- Block Coordinate Gradient Descent (BCGD) with the Gauss-Southwell (GS) Rule
- Randomized Block Coordinate Gradient Descent

The comparison is carried out on both synthetic and real-world datasets (MNIST) using various step size strategies. Results include performance metrics based on accuracy and computational cost (CPU time). All algorithms were implemented in Python using PyTorch and NumPy.

## Authors
- Alessandro Pala - 2107800  
- Tanner Aaron Graves - 2073559  
- Alisa Snezskaia - 2107497  
- Anna Glado - 2122285

## Contents
1. **Algorithms**
   - **Full Gradient Descent**: Standard gradient descent method.
   - **Block Coordinate Gradient Descent (BCGD)**: Optimizes parameters one coordinate or block at a time.
     - Gauss-Southwell rule: Greedily selects blocks with the largest gradient.
     - Randomized rule: Selects blocks randomly.

2. **Step Size Methods**
   - **Fixed Step Size**: Empirical estimation of Lipschitz constant or Frobenius norm.
   - **Line Search**: Exhaustive search along the gradient direction.
   - **Block Step Size**: Inverse of 2-norm of the data matrix block.

3. **Datasets**
   - **Synthetic Dataset**: A high-dimensional dataset with noise generated using normal distribution.
   - **MNIST Dataset**: Flattened grayscale 28x28 images of handwritten digits for classification into 10 classes.

4. **Performance Metrics**
   - **Accuracy**: Classification accuracy across iterations.
   - **CPU Time**: Time taken to achieve convergence.

## Results
### Synthetic Dataset
- Full Gradient Descent consistently outperformed both Gauss-Southwell and Randomized BCGD.
- BCGD methods struggled due to the high-dimensional nature of the problem but showed potential for faster convergence in cases with optimized block selections.

### MNIST Dataset
- The high dimensionality of MNIST data caused BCGD methods, particularly Randomized BCGD, to perform worse compared to Full Gradient Descent.
- Block step size performed better than a fixed step size, although it introduced some instability (high noise in accuracy curves).

