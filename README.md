# JackalML

JackalML is a lightweight, high-performance Machine Learning library integrated directly into the Jackal Programming Language.

Unlike traditional high-level machine learning libraries, JackalML implements its core algorithms natively in C. This design choice provides superior execution speed while preserving a clean and expressive syntax within the Jackal environment.

The library is developed with a strong focus on Explainable AI (XAI) and low-level optimization, making it suitable for academic research, performance-critical systems, and educational purposes.

## Key Features

- Native C Core  
  All mathematical operations and data structures are implemented in C for maximum performance.

- Custom Memory Management  
  Efficient memory handling for large-scale datasets without the overhead of heavy garbage collection.

- No External Dependencies  
  Built entirely from scratch without relying on external libraries such as NumPy or Scikit-learn.

- Academic Ready  
  Supports loss tracking (SSE / Loss) and evaluation metrics suitable for Master's degree (S2) level research.

## Supported Algorithms

### 1. Supervised Learning

- K-Nearest Neighbors (KNN)  
  Distance-based classification using optimized Euclidean distance computation.

- Naive Bayes (Gaussian)  
  Probabilistic classifier based on Gaussian distribution, designed for fast inference.

- Logistic Regression  
  Binary classification using Gradient Descent optimization.

- Decision Tree  
  Native node-based tree structure with recursive construction using Gini Impurity splitting.

### 2. Unsupervised Learning

- K-Means Clustering  
  Partitions observations into K clusters. Includes Sum of Squared Errors (SSE) calculation to support the Elbow Method.

### 3. Data Preprocessing

- Normalization  
  Feature scaling to ensure proportional contribution of each feature.

- Data Shuffling  
  Synchronized shuffling of features and labels to reduce training bias.

## Installation

JackalML is part of the Jackal ecosystem.  
Import the required modules directly in your Jackal source file.

## Usage KNN
```js
import basic
import supervised


let data = [
    [10, 20],
    [20, 30],
    [30, 40],
    [40, 50],
    [50, 40]
]

let labels = [1, 1, 0, 0]

let dataset = DataSet(data, labels)

let knn = Knn(3)
    .fit(dataset)
    .predict([[10, 21]])

println(knn)
```

## Usage Tensor
```js
import Tensor

let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

let t = Tensor<Float>(raw, [20])
                    .reshape([3, 4])
                    .add(5.0)
                    .mul(2.0)
                    .exports()

println(t)
```

