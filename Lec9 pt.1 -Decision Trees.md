# Decision Trees

- Decision tree is a non-linear classifier because the goal is NOT to find a decision boundary.
- It works on making conditions to split values using each feature
  - E.g. if Feature1<8 for samples K, then samples K belong to class 0.

**Example**
Let's find a condition to split feature x1 to be able to predict label y.

| x1 | x2 | y |
|---|---|---|
| 1 | .. | 1 |
| 2 | .. | 1 |
| 3 | .. | 1 |
| 4 | .. | 1 |
| 5 | .. | 1 |
| 6 | .. | 0 |
| 7 | .. | 0 |
| 8 | .. | 0 |
| 9 | .. | 0 |
| 10 | .. | 0 |

- **Condition 1:** if (x1 > 5), then class 0:
  - Probability of class 0 `P(0)`: 1
  - Probability of class 1 `P(1)`: 0
- **Condition 2:** if (x1 > 2), then class 0:
  - Probability of class 0 `P(0)`: 4/6
  - Probability of class 1 `P(1)`: 2/6

We can see that condition 1 is much better than condition 2:
- Condition 1 is **pure**, meaning the probability of one class is 1, while the other is 0.
- Condition 2 is **impure**, meaning the probability of one class is NOT 1, while the other is NOT 0. 

The **best** case is `P(class 1)=1` and `P(class 2)=0`, we call this pure. This gives us the best split.<br>
The **worst** case is `P(class 1)=0.5` and `P(class 2)=0.5`, this is the highest impurity. This gives us the worst split.

## Information Gain
  
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\inline&space;\dpi{110}&space;\huge&space;\color{white}\text{IG}(D_p,&space;f)&space;=&space;I(D_p)&space;-&space;\sum_{j=1}^{m}&space;\frac{N_j}{N_p}*I(D_j)" alt="information gain formula" />
</p>

**Where:**

- $IG$: Information gain
- $I$: Impurity
- $D_p$: Parent
- $D_j$: Child
- $N_i$: Number of samples of child
- $N_p$: Number of samples of parent

The parent splits if impurity of child is smaller.

**We will be working with binary, so this is what the equation will look like:**

<p align="center">
  <img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/f2b95648-5898-4754-9a01-d10148749ef4">
</p>

### What are the measures of impurity (I)?
- Entropy
- Gini Impurity
- Classification Error

## Entropy
<p align="center">
  <img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/900f4372-b722-4a30-a69e-2340939da673">
</p>

**Where:**
- $t$: Subset of class
- $P(i|t)$: Probability of a specific class

Here, we find the probability on different classes.
> We added -ve since $log$ goes from $-inf$ to $0$, and we want it to go from $0$ to $+inf$

**Most pure: 0**<br>
**Most impure: 1**

## Gini Impurity
<p align="center">
  <img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/30be938b-15e3-4530-95c0-d012d987207f">
</p>

- Maximum impurity (worst):
  - $1 - ((0.5)^2 + (0.5)^2) = 0.5$
  - So the maximum value in GI is 0.5
- Minimum impurity (best):
  - $1 - ((1)^2 + (0)^2) = 0$
  - So the minimum value in GI is 0

**Most pure: 0**<br>
**Most impure: 0.5**

## Classification Error
<p align="center">
  <img src="https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/8ff5740f-dd3f-42a4-ac57-848e8a71a9bc">
</p>

- It depends on the probability of each class.
- It's not the best, since it's not sensitive to changes in probabilities in classes and only considers the maximum probability. However the other 2 impurity measures consider all probabilities.

**Most pure: 0**<br>
**Most impure: 0.5**

## Graph of Entropy, Gini Impurity, and Classification Error

![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/72a46585-61d6-4b99-9ef7-05315d7b695b)

## Example

The dataset has 40 samples of class 1 (C1), and 40 samples of class 2 (C2).

We will split the data based on 2 conditions, and use IG on both to find out which split is better.

### **Condition A:**
![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/52957cad-5004-4a72-a25a-ce665699a314)

Now let's calculate the impurity using the 3 measures to be able to get the information gain.

#### Classification Error
- Parent:
  - $I_E(D_p) = 1 - 0.5 = 0.5$
- Left Child:
  - $I_E(D_p) = 1 - 3/4 = 0.25$
- Right Child:
  - $I_E(D_p) = 1 - 3/4 = 0.25$
- Therefore IG will be:
  - $IG(D_p, f) = 0.5 - (\frac{40}{80} * 0.25) - (\frac{40}{80} * 0.25) = 0.25$

#### Gini Impurity
- Parent:
  - $I_E(D_p) = 1 - (0.5^2 + 0.5^2) = 0.5$
- Left Child:
  - $I_E(D_p) = 1 - (\frac{3}{4}^2 + \frac{1}{4}^2) = 0.375$
- Right Child:
  - $I_E(D_p) = 1 - (\frac{1}{4}^2 + \frac{3}{4}^2) = 0.375$
- Therefore IG will be:
  - $IG(D_p, f) = 0.5 - \frac{40}{80} * 0.375 - \frac{40}{80}*0.375 = 1.25$

#### Entropy
- Parent:
  - $I_E(D_p) = -(0.5 * log_2(0.5) + 0.5 * log_2(0.5)) = 1$
- Left Child:
  - $I_E(D_p) = -(\frac{3}{4}*log_2(\frac{3}{4}) + \frac{1}{4}*log_2(\frac{1}{4})) = 0.81$
- Right Child:
  - $I_E(D_p) = -(\frac{1}{4}*log_2(\frac{1}{4}) + \frac{1}{4}*log_2(\frac{1}{4})) = 0.81$
- Therefore IG will be:
  - $IG(D_p, f) = 0.5 - \frac{40}{80} * 0.81 - \frac{40}{80} * 0.81 = 0.19$


### **Condition B:**
![image](https://github.com/SohailaDiab/Supervised-Learning-Course/assets/70928356/5ec02540-8d7c-4091-ae7b-91166a1f04ba)

> We can see that the right child is **PURE**.

#### Classification Error
- Parent:
  - $I_E(D_p) = 1 - 0.5 = 0.5$
- Left Child:
  - $I_E(D_p) = 1 - \frac{2}{3} = \frac{1}{3}$
- Right Child:
  - $I_E(D_p) = 1 - 1 = 0$ 
- Therefore IG will be:
  - $IG(D_p, f) = 0.5 - (\frac{60}{80} * \frac{1}{3}) - (\frac{20}{80} * 0) = 0.25$

#### Gini Impurity
- Parent:
  - $I_E(D_p) = 1 - (0.5^2 + 0.5^2) = 0.5$
- Left Child:
  - $I_E(D_p) = 1 - (\frac{1}{3}^2 + \frac{2}{3}^2) = 0.44$
- Right Child:
  - $I_E(D_p) = 1 - (1^2 + 0^2) = 0$ 
- Therefore IG will be:
  - $IG(D_p, f) = 0.5 - \frac{60}{80} * 0.44- \frac{20}{80}*0 = 0.167$

#### Entropy
- Parent:
  - $I_E(D_p) = -(0.5 * log_2(0.5) + 0.5 * log_2(0.5)) = 1$
- Left Child:
  - $I_E(D_p) = -(\frac{1}{3}*log_2(\frac{1}{3}) + \frac{2}{3}*log_2(\frac{2}{3})) = 0.918$
- Right Child:
  - $I_E(D_p) = -(1*log_2(1) + 0*log_2(0)) = 0$
- Therefore IG will be:
  - $IG(D_p, f) = 0.5 - \frac{60}{80} * 0.918 - \frac{20}{80} * 0 = 0.31$

### Conclusion
- Classification Error:
  - The Information Gain for the split on **condition A** and **condition B** are the **same**.
  - Therefore, classification error is **NOT a good indicator**.
- Gini Impurity:
  - The Information Gain for the split on **condition B** is **HIGHER** than the split on **condition A**.
  - Since higher IG is better, condition B is better.
- Entropy:
  - The Information Gain for the split on **condition B** is **HIGHER** than the split on **condition A**.
  - Since higher IG is better, condition B is better.
  - The IG using Entropy is higher than the IG using Gini Impurity.

> Classification error is the worst option to use in general. It is best to either use Gini Impurity or Entropy.

## When to stop splitting?
> In trees, splitting is training. These conditions that the tree splits on is what it will use in testing/prediction.

There are 3 approaches:
- Stop training when a very small error/cost value is reached.
- Stop when fully converged (error=0).
- Stop when max tree depth is reached.

## How to decide which feature to use?
