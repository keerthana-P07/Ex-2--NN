<H3>Name: P KEERTHANA</H3>
<H3>Register no: 212225230138</H3>
<H3>Date: 28-10-2026</H3>
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
  ```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate
        self.b = 0.0
        self.w = None
        self.misclassified_samples = []

    def fit(self, x, y, n_iter=20):
        self.b = 0.0
        self.w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self.b += update
                self.w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x):
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        return np.where(self.f(x) >= 0, 1, -1)


# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print(df.head())

y = df.iloc[:, 4].values
x = df.iloc[:, :3].values

# 3D Scatter plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("Iris Dataset")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")

ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2],
           color='red', marker='o', s=10, label='Setosa')
ax.scatter(x[50:100, 0], x[50:100, 1], x[50:100, 2],
           color='blue', marker='^', s=10, label='Versicolor')
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2],
           color='green', marker='x', s=10, label='Virginica')

plt.legend()
plt.show()

# Select two features and two classes
x = df.iloc[:100, :2].values
y = y[:100]

# 2D Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(x[:50, 0], x[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1],
            color='blue', marker='x', label='Versicolor')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset - 2D Scatter Plot")
plt.legend()
plt.show()

# Convert labels
y = np.where(y == 'Iris-setosa', 1, -1)

# Standardization
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

# Train Perceptron
classifier = Perceptron(learning_rate=0.05)
classifier.fit(x_train, y_train, n_iter=20)

# Accuracy
y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Error graph
plt.figure(figsize=(6, 4))
plt.plot(range(1, 21), classifier.misclassified_samples, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Perceptron Learning")
plt.show()
```


# OUTPUT:
<img width="587" height="225" alt="image" src="https://github.com/user-attachments/assets/7c7e2e10-7269-4d64-b9af-068d4c4f738a" />
<img width="661" height="547" alt="image" src="https://github.com/user-attachments/assets/d2b0a9df-1eb6-4ecb-883a-a6822d2dc16f" />
<img width="990" height="662" alt="image" src="https://github.com/user-attachments/assets/b067378c-cde5-4bda-bcc8-3c6879801be1" />
<img width="887" height="521" alt="image" src="https://github.com/user-attachments/assets/7f91c452-c5c1-4d6c-936f-105865fe55fd" />





# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
