# Logistic Regression and Backpropagation

## 1. Recall: Logistic Regression

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/89b7c2fe-d7b8-4209-bc59-97d1f01880ff)

This is called **forward propagation**. The inputs are fed into the model, and it moves forward across this architecture and outputs $\hat{y}$ so we can get the prediction and cost. 

### 1.1 💭 Recall: The **Sigmoid activation function** $a$:

<img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/40e7717f-5c0c-4475-a520-ad5b7b1d17a6">

### 1.2 💭 Recall: After getting the $\hat{y}$, we can calculate the **loss** using **Cross-Entropy Loss Function**:

<img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/788b3c32-6e19-4c6a-ad4f-7856e1e5fb4b">

Since $\hat{y} = a$:

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/2a1fb9f4-fccf-4868-8228-83d63fc2fc85)

### 1.3 💭 Recall: This is how weight update (slope) is in Logistic Regression:

<img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/7f643065-4d47-4475-a635-f74c2088116c">

## 2. Backpropagation
- We want to calculate the **derivative of cost $L$ with respect to weights $w$ and bias $b$** to update the weight and bias.
- We do not know what the cost is until the very end, but we want to go back to be able to update the parameters.. how do we do that?
This is where backpropagation comes in.

From above, we know what forward propagation is. In backpropagation, we move to the opposite side so that we can update the weights and bias based on the loss we got.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/92c66701-19bd-4686-bab0-91cd6d54df81)

Using the **chain rule**:
- To be able to get the change in weights ($\triangle{w}$), which is the **derivative of cost $L$ with respect to weights $w$**, this is what we need to calculate:

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/fea0f451-6c74-4607-9ae4-3ade2100614c)

Using the **chain rule**:
- To be able to get the change in bias ($\triangle{b}$), which is the **derivative of cost $L$ with respect to bias $b$**, this is what we need to calculate (same as above, just change $w$ to $b$):

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/2a5595f3-493a-4ea4-ab0d-769ee8cd457b)

### 2.1. Let's see how to calculate it 🧮

### 2.1.1 $\frac{\partial{L}}{\partial{a}}$:

> Refer to the Cross-Entropy Loss Function above.

First, recall that the derivative of $log_e(x)$ $=$ $\frac{1}{x} \cdot {\text{derivative of x}}$

$\frac{\partial{L}}{\partial{a}}$ is the derivative of loss with respect to a (activation/output)

Therefore,


![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/73f29f4a-b004-4661-95fd-86498622eaed)

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/7d2d2eb3-47fc-4592-ae2b-f04850aca508)

### 2.1.2 $\frac{\partial{a}}{\partial{z}}$:
$\frac{\partial{a}}{\partial{z}}$ is the derivative of output/activation with respect to z

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/1292566e-8122-4bdd-96d5-2dba5515654b)

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/3fbef096-839e-4796-9ffc-13bb1f529f9f)

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/f20873df-26ef-4e4f-9d0d-cf9a322bea6c)

As we can see, $\frac{1}{(1- e^{-z})}$ is the **Sigmoid function**, so we can rewrite it as:

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/24813d95-9c72-4b2b-8582-a20eb5869f81)

### 2.1.3 $\frac{\partial{z}}{\partial{w}}$:
$\frac{\partial{z}}{\partial{w}}$ is the derivative of z with respect to w (weights)

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/dffd1fc8-2486-4aeb-970d-cc7927eff432)

### 2.1.4 $\frac{\partial{z}}{\partial{b}}$:
$\frac{\partial{z}}{\partial{b}}$ is the derivative of z with respect to b (bias)

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/60afd5d7-4a93-4436-b9b8-54887f7c36d0)

### 2.2. Calculating $\frac{\partial{L}}{\partial{w}}$ and $\frac{\partial{L}}{\partial{b}}$

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/2959471c-5f25-47d4-a317-92abe45af324)

- We know that $\hat{y} = a$
- So, just like we wanted, this derivative gives us the weight update!
- $\triangle{w} = \eta (a-y)x$

![png](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/650443ff-0948-4242-8bbb-32a8009f98e0)

- Again, we know that $\hat{y} = a$
- So, just like we wanted, this derivative gives us the bias update!
- $\triangle{b} = \eta (a-y)$
