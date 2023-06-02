# Regularization
Sometimes, our model **overfits** the data (high variance). This means that it memorized the training set and got a high accuracy while training, but it does not generalize/perform well on unseen data (testing set).

Basically:<br>**training set** -> high accuracy<br>**testing set** -> low accuracy

![image](https://user-images.githubusercontent.com/70928356/236968666-c088ef32-8ca4-40bd-824a-cc43e4c0fa7a.png)

Our goal is to have our model **GENERALIZE WELL ON UNSEEN DATA**

> Note: Overfitting does not only mean that we have **100%** accuracy on training and a much lower accuracy on testing; having a 90% accuracy in training and 80% accuracy in testing is also considered to be overfitting.

**Two of the many ways to reduce overfitting:**
- Collect more data
- Introduce a **regularization parameter** (λ)
  - L1 regularization method
  - L2 regularization method

## L1 Regularization (also called Lasso)
**The cost function we all know:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}J(w,b)%20=%20\frac{1}{2m}%20\sum_{i=1}^{m}%20(\hat{y}_i%20-%20y_i)^2" style="background-color:black;" alt="J(w,b)" />

Where: $\hat{y} = z = w^\top x$

*Our overall goal is to get the w and b that **minimize the cost***

- In order to achieve L1 regularization, we will have to make changes to the cost function. 
- We will do this by adding the `λ` parameter multiplied by `|w|`. 
- `|w|` is the summation of the absolute of all the weights we have.

**L1 regularization:**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}J(w,b)%20=%20\frac{1}{2m}%20\sum_{i=1}^{m}%20(\hat{y}_i%20-%20y_i)^2%20+%20\frac{\lambda}{2m}%20\sum_{j=1}^{m}%20|w_j|" style="background-color:black;" alt="J(w,b) L1"/>

- We divided `λ` by `2m` to for normalization.

*(Will explain soon why and how this affects the cost function and helps reduce overfitting)*

## L2 regularization (also called Ridge)
- Instead of taking the absolute value of `w` (weights), it takes it **squared**.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}J(w,b)%20=%20\frac{1}{2m}%20\sum_{i=1}^{m}%20(\hat{y}_i%20-%20y_i)^2%20+%20\frac{\lambda}{2m}%20\sum_{j=1}^{m}%20w_j^2" style="background-color:black;" alt="J(w,b)"/>

**OR**

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}J(w,b)%20=%20\frac{1}{2m}%20\sum_{i=1}^{m}%20(\hat{y}_i%20-%20y_i)^2%20+%20\frac{\lambda}{2m}%20\|w\|_2^2" style="background-color:black;" alt="J(w,b)"/>

- `||w||` is called the **norm**. 
- Norm basically the length of the vector `w.T * w`, which returns a scalar value (a number, not a vector).
- It is the same as summing all the weights `w` squared.

## Now what we know the formulas for L1 and L2, what effect do they have on the weights? 🤔
⬇ They cause **weight decay** (or **weight shrinkage**), which means the weight decreases. ⬇

### What happens when weights decrease?
> Recall: $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2\$

- We know that the goal is to minimize $J(w,b)$, the cost. From the formula above, we can see that the cost depends on $\hat{y}$, which depends on $w$, since $\hat{y} = w^\top x$
- Therefore, if we decrease the dependency on $w$ _(which means decrease/shrink $w$)_, it will affect $\hat{y}$, leading to the cost being affected as well.

**Let's take a look at 1 feature:**
$w_{(1)} . x_{(1)}$

- When $w_{(1)}$ decreases, it reduces how the model is dependant on feature $x_{(1)}$
- **So, what happens when $w_{(1)}=0$?**: This means that the model is not dependant at all on feature $x_{(1)}$, which means it doesn't use it to predict.

### How does weight decay affect the model?
It makes the model smoother, which reduces overfitting as it will not memorize the features with their exact values in the dataset.

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/8d25705f-bf55-4759-94d4-10c006731575)

Therefore, we want to be able to reduce dependency on some weights to be able to regularize. We want to shrink the weights, having them approach 0.

> **📝 TLDR: The goal of weight decay is to shrink/decrease the weights $x$, which reduces the dependency and contribution of features $x$ on the model. This will help the model not memorize the features, which reduces overfitting.**

⚠ **Sure, this may decrease the training accuracy, for example, from 99% to 96%, but in return the model will be able to _generalize and predict well on unseen data_ (test set). Therefore, the testing accuracy will increase.** ⚠


## HOW does L1 and L2 affect the weights and decrease overfitting? 🤔
✨ Basically, introducing the `λ` (lambda) to the cost function is what makes it happen. ✨

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\color{white}J(w,b)%20=\underline{\color{red}\frac{1}{2m}%20\sum_{i=1}^{m}%20(\hat{y}_i%20-%20y_i)^2}%20+\underline{\color{blue}\frac{\lambda}{2m}%20\sum_{j=1}^{m}%20|w_j|}" style="background-color:black;" alt="J(w,b) L1"/>

> *(This is Ridge, but the same concept applies to Lasso as well.)*
> Also, keep in mind that `λ` is a hyperparameter that we set!

- Again, our goal is to minimize the cost (the part in red) $\frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2\$, in order to minimize the whole cost function $J(w, b)$.
- We will make use of **adding to the cost**, still holding on to the objective of minimizing the cost function.


- As we **increase `λ`**, the whole term after the `+` sign (in blue) also increases. Obviously, this leads to the **cost to increase** as well. 😬
- Since we want to minimize the cost, when `λ` increases, it will force the weights `w` to decrease to **maintain a low cost**.

### `λ` ⬆ `|w|` ⬇

So, as we explained before, when the weights decrease the dependency on some features decrease as well. Therefore, the model will not heavily rely on these features and memorize them, leading to a smoother curve that generalizes better on unseen data.

## Lasso vs. Ridge 🧐
|            | L1 Regularization                                     | L2 Regularization                                     |
|------------|-------------------------------------------------------|-------------------------------------------------------|
| Definition | Adds the absolute value of the coefficients to the loss function | Adds the squared magnitude of the coefficients to the loss function |
| Effect     | Encourages sparsity (i.e., some coefficients become exactly zero) | Encourages small but non-zero coefficients              |
| Feature Selection | Can be used for feature selection as it tends to produce sparse solutions | Doesn't perform feature selection, but reduces the impact of less important features |
| Interpretability | Leads to more interpretable models with a smaller set of relevant features | Does not explicitly result in a smaller set of relevant features |
| Computational Complexity | More computationally expensive due to the non-differentiability at zero | Less computationally expensive due to differentiability at all points |
| Robustness | More robust to outliers and irrelevant features due to sparsity | Less robust to outliers, but still reduces the impact of less important features |
| Solution | May result in solutions with fewer variables | Produces solutions with all variables, but with smaller coefficients |
| Multicollinearity | May have difficulties in the presence of highly correlated features | Handles multicollinearity better due to smaller coefficient magnitudes |

## Extra References:
- [When should I use lasso vs ridge? | Stack Exchange](https://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge)
- [Ridge vs Lasso Regression, Visualized!!! | StatQuest Video](https://www.youtube.com/watch?v=Xm2C_gTAl8c)
- [Regularization in Machine Learning | Towards Data Science](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
