# Multi-Class Classification
So far we have worked on binary classes only. So, how do we deal with classes that are more than 2?

There are two ways we can perform multi-class classification:
- One vs. One
- One vs. Rest

## 1. One vs. One
- Here, all possible permutations between the different classes are considered. For each permutation, there will be a different classfier.
- This is not easy and is very time consuming, since the classification is repeated for each permutation.. everything from initializing weights, iterating and optimization, prediction, etc. will be repeated for each permutation.

If we have 3 classes, it would be something like this (where the class on the left is 1 and the one on the right is 0):
- Class 1 vs. Class 2
- Class 1 vs. Class 3
- Class 2 vs. Class 1
- Class 2 vs. Class 3
- Class 3 vs. Class 1
- Class 3 vs. Class 2

As the number of classes increases, this method will be veeeery inefficient.

## 2. One vs. Rest 
- Also called One vs. All
- Here, for each class, one it will have a positive label (1), while all the other class will have a negative label (0).
- There will be n classifiers, where n is the number of classes.

If we have 3 classes, it would be something like this:

|          | Class 1 | Class 2 | Class 3 |
|----------|----------|----------|----------|
| Classifier 1 (C1)   |     1     |      0    |     0     |          
| Classifier 2 (C2)   |      0    |       1   |      0    |          
| Classifier 3 (C3)   |       0   |        0  |       1   | 

## 3. How do we make multiple classifiers?

- First, we need to on-hot encode the labels.
  - **Why one-hot encoding?**<br>Because we need the distances between the different labels to be the same. For example, if we have classes 1, 2 and 3, the distance between class 1 and 2 will be 1, while the distance between 1 and 3 will be 2, which does not make sense. The labels can also be non-numeric.

**Normally, we use to have this:**

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/0b557bb4-3de1-4f2c-950d-022e6d993dc6)

**To turn it to multi-class classification we will do this:**

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/9d7187d8-c163-4480-956c-07667f2303d7)

- Since we have 3 classes here, we need 3 classifiers.
- We will no longer have a weights vector, we will have a **weights matrix** where the columns are the weights for each classifier, and the rows are the features.
- Here, we have weights that are unique to each classifier. To access the weights of the first classifier, we only need to do `w[:, 0]`.
- The **bias** is only 1 value for each classifier, so now we will have a vector of biases `1 x C`, where `C` is the number of classes. In this example we will have 3 bias values.


## 3.1 Okay now that we have the weights matrix and bias vector ready, what do we do?
Just like normal training, we will have epochs.

- Inside the epochs loop, we will loop over each **column** in the **weight matrix**, where we will perform gradient descent. 
  - Each column of the weight matrix, let's say column 1, means that all labels are 0 except the label for class 1, which is 1.
- **Taking a look inside the first iteration (first classifier)**:
  - We will take the **X matrix** _(features x samples)_ which contains the features as it is.
  - Since at the very beginning we performed one-hot encoding on the target variable **y**, the dimensions of matrix y are _(samples x classes)_.
  - In this iteration, we will take the **first column of y** `y[:,0]` only.
  - This is because in the FIRST iteration, we are **TRAINING THE FIRST CLASSIFIER**. This means we will be working on the **first class**, where the labels for class 1 is 1 and any other class is 0.
- After finishing iterating on the number of epochs, where in each epoch we were iterating on each column of the weight matrix (training based on the respective classifier), we will end up with predictions that are a **matrix**, where the dimensions are _(samples, classes)_.

But wait.. we are not done yet! We got the z values (output matrix), but the numbers are all over the place.. there are huuuge and small numbers, both positive and negative; how do we interpret them?? 

## 3.2 How to interpret the output? 🤔
Let's say we have 3 classes: R, G and B. Using what we mentioned above, we will have 3 binary classifiers each predicting different output values (the output values are linear (-∞, ∞)).

**Before prediction, our one-hot encoded true labels look something like this:**

| R | G | B |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 0 | 0 | 1 |

**And, let's say this is the model's output values/predictions:**

| R | G | B |
|---|---|---|
| 90 | 10 | -10 |
| -100 | -1 | ... |
| ... | ... | ... |
| ... | ... | ... |

**How to make sense of these values to be able to decide the predicted label?** 😵

We can't just take the maximum value and say that our prediction belongs to that class, because it doesn't make sense. It doesn't tell us how correct that prediction is, or to what extent does the model think that this is the true label. Therefore, we need to convert these z values to probability to find out the **probability of sample X belonging to class Y**.

Thankfully, the exponential function comes to the resue!

### 3.2.1 Introducing Softmax

<p align="center">
  <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}&space;\huge&space;\color{white}\text{softmax}(z_i)&space;=&space;\frac{e^{z_i}}{\sum_{i=1}^{n}&space;e^{z_i}}" alt="equation" />
</p>

Where:
- $z_i$: The predicted label of sample $i$.

ℹ **Softmax finds the probability of the predicted value belonging to a class C. We will then pick the class corresponding to the highest probability as the predicted class.** ℹ

**After applying Softmax, the predictions (z matrix) will approximately look like this:**
| R | G | B |
|---|---|---|
| 0.9 | 0.1 | -10 |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |

- So for this first sample, the predicted class will be R since it has the highest probability
- As you can see, the **sum of the values over the classes is equal to 1**. It will always be equal to one for each sample.

#### **What does $e^{z}$ do?**
- When z is -∞, the value will be 0
- When z is ∞, the value will be 0

Therefore, there will be **no negative** values.

#### **Why do we not want negative values?**
- Because when there are no negative values, we will be able to sum the values to get a probability.<br>
- Eg. $\frac{e^{90}}{e^{900}+e^{10}+e^{-10}}$.<br>
- $e^{-10}$ will be a very small non-negative number.

#### Why not just eliminate any negative numbers by making it 0?
- Because that means we will discard any info the negative numbers give us. 
- What if all z values are negative? How will we pick the predicted label if they are all the same value?

### 📝 To sum it up
1. Get z (pred) values for each sample, for each class.
2. Get Softmax of z over each sample for n classes with this equation:

<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{80}&space;\huge&space;\color{white}\text{softmax}(z_i)&space;=&space;\frac{e^{z_i}}{\sum_{i=1}^{n}&space;e^{z_i}}" alt="equation" />

3. Softmax will output probabilities
  - Each sample will have C probabilites, where C is the number of classes. 
  - The probability means "The probability that sample i belongs to class C".
  - The probabilities per sample over the classes will have the sum of 1.
4. The class corresponding to the highest probability will be the predicted class for sample i.

## Code
```py
```
