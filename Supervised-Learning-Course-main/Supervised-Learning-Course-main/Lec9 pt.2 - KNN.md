# KNN - K Nearest Neighbors

- A **non-linear**, **classification** algorithm.
- It looks at the neighboring points' labels to decide what the label to assign the new example.
- **K** is a hyperparameter, which how many neighboring points will the new example depend on.

**Let's take k=3 as an example:**
- We will go over the **WHOLE** training set, and calculate the distance between each point and the new example.
- Then, take the 3 _(since k=3)_ training samples that had the smallest distance between them and the new example.
- This new example's predicted label will be the majority class out of the 3 training samples we got.
  - For example, if the 3 neighboring training samples had the classes {⭐, 🔺, 🔺}, then the predicted label will be 🔺.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/ca9135cd-b436-4ff1-a9e2-34ddcd61c268)

## How to calculate the distance?
Calculate it by using the **Eculidean Distance**

<p align="center">
  <img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/730ab6fc-9b68-47f5-b8e2-53c75742f0a5">
</p>

### If we have 2 classes $x$ and $y$:
<p align="center">
  <img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/c63e3c4a-963a-4aa8-a51e-391a885a44e3">
</p>

**Where:**
- $x = x_{point 2}-x_{point 1}$
- $y = y_{point 2}-y_{point 1}$

## Algorithm
- Iterate over testing (unseen) samples
  - **Inside the testing dataset iteration:**
  - For each testing sample, iterate over the **whole** the training dataset.
    - **Inside the training dataset iteration:**
    - Calculate the Eculidean Distance between the current test sample and current train sample.
    - Add this distance to a vector _(with the size of the number of training samples)_, where the index of the distance maps to the index of the training sample.
  - Sort the distance vector in ascending order, keeping the indices in another column to be able to map back to that training sample.
  - Get the first **K** training samples in the sorted distance vector.
  - See what label these **K** training samples map to.
  - Sum up the number of each class (frequency).
  - The label assigned/predicted for this test sample will be the label with the highest frequency. (If there are 2+ classes with the same frequency, pick a random one)
  - Store the label in a matrix to correspond with the test sample index.

## Important Notes

- As **K** increases ⬆:
  - The boundaries will be smoother.
  - There will be less variability.
  - ⚠ A high K may result in **UNDERFITTING**, causing **high bias**.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/4a3df8f7-df7d-4c5e-8fb7-05fe22e5dd0d)

- As **K** decreases ⬇:
  - The boundaries will be more rough.
  - There will be more variability.
  - ⚠ A low K may result in **OVERFITTING**, causing **high variance**. _eg. k=1 is very bad since it basically memorizes the data_

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/d34c2b1d-3847-4578-9c46-d066c2175bef)

<br>
- The number of classes should not control K.
  - We do not want 2 parameters to be dependant on each other.
  - We achieve orthogonalization when no parameters are dependant on one another, which is what we want. 
