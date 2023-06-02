# Optimizers
- The goal of optimizers is to **converge faster**.
- Optimization is a process that is used with the goal to minimize the loss function and improve the model performance.
- To do that, it adjusts (increase/decrease) the weights and other parameters to find the values that result in the minimum loss.

## 1. Gradient Descent (Batch GD)
![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/0e08cb4d-46c2-4864-9001-2c3a3d97ba6f)

- The goal of Gradient Descent is to minimize the loss function (same as all other optimizers).
- It iteratively adjusts the model's parameters by following the negative gradient of the cost function to reach the minimum.

### How does it work? 🛠
- First, we initialize the parameters $w$ (weights) and $b$ (bias) randomly. This means we will start from a random point on the cost function curve.
- Then, we will compute the **derivatives of the loss function**, with **respect to $w$** and with **respect to $b$**.
- The weights and bias are updated using the whole data at once (can be also called **whole batch**).

**Let's take the MSE loss function for this example:**
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

**Derivative of the loss function with respect to $w$:**
$$\frac{{\partial J}}{{\partial w}} = d_w = \frac{-2}{{m}} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$

**Derivative of the loss function with respect to $b$:**
$$\frac{{\partial J}}{{\partial b}} = d_b =  \frac{-2}{{m}} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

$where: \hat{y}^{(i)} = w^{(i)} \cdot x^{(i)} + b$ (This is our predicted value, we took linear regression in this example)

- We will then **update the $w$ and $b$** by moving in the **opposite direction of the slope** increase from the current point by the computed amount multiplied by the learning rate (how big the step we take)

$$w = w - \eta \cdot d_w$$

$$b = b - \eta \cdot d_b$$

- 🔁 Keep iterating until done with epochs or if cost stops increasing/reached local minima.

<details><summary> **Click here for Gradient Descent implementation on Linear Regression!🛠** </summary>
  
```py
  def train_gradient_descent(self, X, y):
    '''
      Inputs : X, y
      Output : m (weights), b (bias)
    '''
    # Initialize weights and biases
    self.m = random.uniform(0, 1)
    self.b = random.uniform(0, 1)
    n = len(X)

    for i in range(self.epochs):
      # The derivatives with respect to m and b
      d_m = (-2/n)*np.sum( (y - (self.m*X + self.b))*X )
      d_b = (-2/n)*np.sum(y - (self.m*X + self.b))

      # Update the parameters (m and b)
      self.m = self.m - self.lr*d_m
      self.b = self.b - self.lr*d_b

    return self.m, self.b
```
</details>
  
## 2. Mini-Batch Gradient Descent
- It is a variant of gradient descent, where the training data is divided into small subsets called mini-batches.
- The model's parameters ($w$ and $b$) are updated using the gradient of a mini-batch. It calculates the average gradient of the cost function for the mini-batch and updates the parameters in the opposite direction.

### How does it work? 🛠
This is done **per epoch**:
- We take a mini-batch.
- Compute the **derivatives of the loss function**, with **respect to $w$** and with **respect to $b$**.
- The weights and bias are updated using the gradient we just calculated.
- 🔁 Repeat for each mini-batch in the training set.

<details><summary> **Click here for Mini-Batch Gradient Descent implementation on Logistic Regression!🛠** </summary>
  
```py
def train_mbgd(self, X, y, batch_size):
    '''
      Inputs : X, y, batch_size
      Outputs : m (weights), b (bias)
    '''
    # Initialize weights and biases
    self.m = random.uniform(0, 1)
    self.b = random.uniform(0, 1)
    
    # Calculate number of batches
    num_samples, num_features = X.shape
    num_batches = num_samples // batch_size
        
    # Iterate over num of epochs
    for _ in range(self.epochs):
    
      # Shuffle dataset every epoch
        shuffle = np.random.permutation(num_samples)
        X_shuffled = X[shuffle]
        y_shuffled = y[shuffle]
        
        # Loop over each batch and train
        for j in range(num_batches):
          batch_start = j * batch_size
          batch_end = batch_start + batch_size

          X_batch = X_shuffled[batch_start:batch_end]
          y_batch = y_shuffled[batch_start:batch_end]

          z = np.dot(X_batch, self.w.T) + self.b
          y_pred = self.sigmoid(z)

          # Find the gradient by getting the derivatives of w and b 
          d_w = np.dot(X_batch.T, (y_pred - y_batch))
          d_b = np.mean(y_pred - y_batch)

          # Update the parameters (m and b)
          self.w = self.w - self.lr*d_w
          self.b = self.b - self.lr*d_b

    return self.m, self.b
```
</details>

## 3. Stochastic Gradient Descent
> In Batch Gradient Descent we were considering all the examples for every step of Gradient Descent. But what if our dataset is very huge? Models crave for data; the more the data the more chances of a model to be good. Suppose our dataset has 5 million examples, then just to take ONE step the model will have to calculate the gradients of all the 5 million examples!! This does not seem an efficient way. To tackle this problem we have Stochastic Gradient Descent.

- It is a variant of gradient descent.
- Here, we update the parameters $w$ and $b$ using the gradient of **one training sample at a time**.
- It **randomly selects a training example**, computes the gradient of the cost function for that sample, and updates the parameters in the opposite direction of the gradient.
- Computationally **efficient** and can converge faster than batch gradient descent. 
- It can be **noisy and may not converge** to the global minimum.

### How does it work? 🛠
This is done **per epoch**:
- Same steps as (batch) gradient descent.
- We take one random sample.
- Compute the **derivatives of the loss function**, with **respect to $w$** and with **respect to $b$**.
- The weights and bias are updated using the gradient we just calculated.
- 🔁 Repeat for all examples in the training set.

<details><summary> **Click here for Stochastic Gradient Descent implementation on Linear Regression!🛠** </summary>

```py
def train_SGD(X, y, learning_rate, num_epochs):
    num_samples, num_features = X.shape
    weights = np.random.randn(num_features)  # Initialize weights with random values
    bias = np.random.randn()  # Initialize bias with a random value

    for epoch in range(num_epochs):
        # Shuffle the training data
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(num_samples):
            # Select a single example
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]

            # Compute the predicted output
            y_hat = np.dot(x_i, weights) + bias

            # Compute the gradients
            gradient_w = x_i * (y_hat - y_i)
            gradient_b = y_hat - y_i

            # Update the model parameters
            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b

    return weights, bias
```
</details>

## 4. Batch vs Mini-Batch vs Stochastic Gradient Descent

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/1279d14f-5595-4c23-bb7e-c3f7353ce30c)

|                     | Batch GD     | Mini-Batch GD        | Stochastic GD   |
|---------------------|--------------|----------------------|-----------------|
| Data Processing     | Entire dataset | Mini-batches         | Single example  |
| Computational Cost  | High         | Moderate             | Low             |
| Convergence         | Slower       | Faster than batch    | Faster than batch|
| Stability           | More stable  | Moderate stability   | Less stable     |
| Noise in Updates    | N/A          | Moderate noise       | High noise      |

- Stochastic Gradient Descent has very noisy updates, since the parameters are updated after one sample, so the variability is very high.
- On the other hand, Batch GD is the smoothest, since it updates parameters after all of the samples. However, it needs a lot of computational power and it is not suitable for large datasets.
- Mini-Batch is the most commonly used out of Batch and Stochastic GD, since it combines the advantages of both.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/847b03c4-09c5-4981-b96e-dd8b4da5cec1)


## 5. Momentum
### 5.1 First of all.. let's see what's an Exponentially Weighted Average is
- Exponentially Weighted Average is a method to calculate a weighted average of previous data points, giving more weight to recent values and less weight to older values.
- More weight is given to point at time $t-1$ than the current point at time $t$.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/566fd2b9-901a-4de9-a8be-a1d87573b082)

In the image above:
- In the red curve, the points are NOT dependant on their previous points. This leads to high variability and zigzag curve, since the points doesn't see any other point but itself.
- In the blue curve, the points depend on the point before it. Let's say by 90% as an example. This means it depends on itself by 10% only. 
- If the point depended on the previous point by 100%, the current point will be exactly like the previous point.

More about EWA:
- It is commonly used to smooth out noisy data and capture underlying trends or patterns.
- The weights assigned to each data point decrease exponentially as you go further in time; this is why it is called Exponentially Weighted Average.
- The amount of smoothing or weighting applied to each data point is controlled by a smoothing factor or parameter, often denoted as "alpha" $α$. This $α$ is the **90%** mentioned in the bullets above.

$$V_t = \beta \cdot V_{t-1} + (1 - \beta) \cdot \theta_t $$

- $V_t$: Current average
- $\beta$: Hyperparameter that is between 0 and 1
- $V_{t-1}$: Previous average
- $\theta_t$: Current data point

> NOTE: If you want more details, check out the first YouTube video linked down in Extra References.

### 5.2 What does Exponentially Weighted Average have to do with optimization?
It introduces **momentum** concept.

### 5.3 Momentum
- It is based on the Exponentially Weighted Average.
- It also depends on previous steps, which means we look at the previously updated weight. This affects the current weight.

**Normal weight update with GD:**
$$w_{new} = w_{old} - \triangle w$$

**Weight update with Momentum:**

$$V_{dw} = \beta \cdot V_{dw} + (1 - \beta) \cdot dw$$

Where:
- $V_{dw}$: Previous time point
- $\beta$: Hyperparameter that controls dependency on previous point. Usually `0.9`
- $dw$: Weight update $\triangle w$, which is at current time point `t`

**Example:**
- We want to depend on the previous weight by 90%.. so the formula will look like this:
$$V_{dw} = 0.9 \cdot \triangle w_{t-1} + (1 - 0.9) \cdot \triangle w_{t}$$

- $0.9 \cdot \triangle w_{t-1}$: Here, we took **90% of the previous weight update**.
- $(1 - 0.9) \cdot \triangle w_{t}$: Here, we took **10% of the current weight update**.

✔ This makes weight updates smoother ✔

### 5.4 Eliminate bias
⚠ **However, we have a problem!** ⚠
- There is bias since we heavily rely on points close to the current. The current points can see the previous points, but previous points cannot see the current point.

✅ We can eliminate bias by dividing $V_{dw}$ by $1-\beta^{t}$, so now $V_{dw}$ looks like:

$$V_{dw} = \frac{\beta \cdot V_{dw} + (1 - \beta) \cdot dw}{(1 - \beta^{t})}$$

### 5.5 How do we use $V_{dw}$ to update weights? 🛠
1. Calculate the change in weights:<br> 
dw = $\triangle w$
2. Use the $V_{dw}$ equation:<br> 
$V_{dw} = \frac{\beta \cdot V_{dw} + (1 - \beta) \cdot dw}{(1 - \beta^{t})}$
3. Update the weights:<br>
$w = w - \eta \cdot V_{dw}$

> This new weight will help with achieving smoother updates.

### 5.6 How do we use $V_{db}$ to update bias? 🛠
It is the same formula and steps as updating the weights. The only difference is that we will use $\triangle b$ instead.

$$V_{db} = \frac{\beta \cdot V_{db} + (1 - \beta) \cdot db}{(1 - \beta^{t})}$$

1. Calculate the change in bias:<br> 
db = $\triangle b$
2. Use the $V_{db}$ equation:<br> 
$V_{db} = \frac{\beta \cdot V_{db} + (1 - \beta) \cdot db}{(1 - \beta^{t})}$
3. Update the weights:<br>
$b = b - \eta \cdot V_{db}$

### 5.7 The effect of the learning rate $\eta$
The learning rate ($\eta$) helps to take only a part of $V_{db}$ and $V_{dw}$, since with time $V_{db}$ / $V_{dw}$ can inflate/explode (increase greatly); so we will only take part of it to avoid overshooting.

### 5.8 How does momentum help with jumping over local minima?
- Momentum gives you a push. 
- At points where the gradient is very steep, it will take faster steps, which makes it skip the local minima. When it's steep, the slope does not change/changes veeery slowly, and there is not much difference in the slope of 2 points that are close to each other.
- At points where the gradient is not steep, it will have slower updates. Here, the gradient of two points next to each other may vary a lot.

> So in general, if there is not much of a difference between the slope of the current point and the previous point, the weights will be updates faster.

🛑 **Training will stop when it reaches slope 0 (db = 0)** 🛑

## 6. RMSprop (Root Mean Squared Propagation)
- Uses 2nd momentum ($\triangle w$ and $\triangle b$ are squared)
- Same concept for momentum
- The algorithm calculates an exponentially weighted average of the squared gradients for each parameter.
- It divides the learning rate by the square root of this average, resulting in larger updates for parameters with smaller gradients and smaller updates for parameters with larger gradients.
- Effective in scenarios with sparse gradients or in problems with non-convex loss surfaces.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/4ea1857a-ce50-4975-b0bf-725d5fc47063)

### 6.1 How do we use $V_{dw}$ to update weights? 🛠
It is the similar to momentum. However, the **change in weights is squared**, and the **updates will be done by multiplying the learning rate with 1/square root of V_dw**.

$$V_{dw} = \frac{\beta \cdot V_{dw} + (1 - \beta) \cdot {dw}^2}{(1 - \beta^{t})}$$

1. Calculate the change in bias:<br> 
dw = $\triangle w$<br><br>
2. Use the $V_{dw}$ equation:<br> 
$V_{dw} = \frac{\beta \cdot V_{dw} + (1 - \beta) \cdot {dw}^2}{(1 - \beta^{t})}$<br><br>
3. Update the weights:<br>
$w = w - \eta \cdot\frac{1}{\sqrt{ V_{dw}}}$<br><br>


### 6.2 How do we use $V_{db}$ to update bias? 🛠
It is the similar to momentum. However, the **change in bias is squared**, and the **updates will be done by multiplying the learning rate with 1/square root of V_db**.

$$V_{db} = \frac{\beta \cdot V_{db} + (1 - \beta) \cdot {db}^2}{(1 - \beta^{t})}$$

1. Calculate the change in bias:<br> 
db = $\triangle b$<br><br>
2. Use the $V_{db}$ equation:<br> 
$V_{db} = \frac{\beta \cdot V_{db} + (1 - \beta) \cdot {db}^2}{(1 - \beta^{t})}$<br><br>
3. Update the weights:<br>
$b = b - \eta \cdot\frac{1}{\sqrt{ V_{db}}}$<br><br>

## 7. Adam (Adaptive Momentum)
- One of the strongest and most widely used optimizers.
- It combines both **RMSprop** and **Momentum**.
- Takes the update of momentum and the update of RMSprop to one equation.

### 7.1 Weight update
$$w = w - \eta \cdot \frac{V_{dw}}{\sqrt{V_{dw}}}$$

### 7.2 Bias update
$$b = b - \eta \cdot \frac{V_{db}}{\sqrt{V_{db}}}$$

### 7.3 How does dividing by $\sqrt{V_{dw}}$ help?

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/ebf38b40-54e2-45e6-8dd2-2e11cbf484a2)
- To take larger, faster steps and converge quicker, we want to stretch the updates as shown in the image above.
- We can see that to stretch, we increase the weights and decrease the bias.
- We do that by dividing the learning rate by $\sqrt{V_{dw}}$.

We know that the bias $b$ is bigger than the weights $w$, as seen in the image above.

Let's take an example to see how this works. 
- Let's say bias is 0.1, while weight is 0.01.
  - When dividing by 0.1, it is like multiplying by 10
  - However, when diving by 0.01, it's like multiplying by 100!
  
💪 **Therefore, the weights increase while the bias decreases, giving us a faster, more stretched updates that converge faster!**

## 8. Comparing the optimizers 🔎
| Algorithm                  | Description                                                                                   | Pros                                                                                        | Cons                                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Batch Gradient Descent     | Updates model parameters after evaluating the gradients over the entire training dataset.    | Guaranteed convergence to a global minimum.                                                 | Computationally expensive for large datasets.                                                         |
| Mini-Batch Gradient Descent| Updates model parameters using a subset (mini-batch) of the training dataset.                 | Faster convergence than batch gradient descent.                                            | Requires tuning of mini-batch size.                                                                  |
| Momentum                   | Uses an exponentially weighted average of past gradients to accelerate convergence.            | Helps to escape local minima and accelerates convergence.                                    | Can overshoot and oscillate around the minimum.                                                       |
| RMSprop                    | Adapts the learning rate for each parameter based on the average of squared gradients.        | Effective in dealing with sparse gradients and non-convex loss surfaces.                   | Requires tuning of hyperparameters.                                                                  |
| Adam                       | Combines the benefits of momentum and RMSprop by using adaptive learning rates and gradients. | Fast convergence, works well in practice for a wide range of problems.                      | Computationally more expensive than basic optimization algorithms. Requires tuning of parameters.  |


## Extra References
- [Exponentially Weighted Moving Average or Exponential Weighted Average | YouTube](https://www.youtube.com/watch?v=XV1f_srZg_E)
- [Batch, Mini Batch & Stochastic Gradient Descent | Towards Data Science](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a)
- [How Does the Gradient Descent Algorithm Work in Machine Learning? | Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/)
- [RMSProp (C2W2L07) | YouTube (Andrew Ng)](https://www.youtube.com/watch?v=_e-LFe_igno)
- [RMSProp | Dive Into Deep Learning](http://d2l.ai/chapter_optimization/rmsprop.html)
- [A Complete Guide to Adam and RMSprop Optimizer | Medium](https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be)
