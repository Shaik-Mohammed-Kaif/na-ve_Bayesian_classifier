# Naïve Bayes Theorem and Feature Formulas

## 1. Naïve Bayes Theorem
The Naïve Bayes classifier is a probabilistic model based on **Bayes’ theorem**, assuming independence among features.

```math
P(Y | X) = \frac{P(X | Y) P(Y)}{P(X)}
```

Where:
- `P(Y | X)`: Posterior probability (Probability of class `Y` given feature vector `X`)
- `P(X | Y)`: Likelihood (Probability of `X` given class `Y`)
- `P(Y)`: Prior probability (Probability of class `Y` before observing features)
- `P(X)`: Evidence (Probability of `X` across all classes)

Since `P(X)` is constant for all classes:

```math
P(Y | X) \propto P(X | Y) P(Y)
```

## 2. Naïve Bayes Assumption
The model assumes each feature is conditionally independent given the class label `Y`:

```math
P(X | Y) = P(x_1, x_2, ..., x_n | Y) = P(x_1 | Y) P(x_2 | Y) ... P(x_n | Y)
```

Thus, the simplified formula becomes:

```math
P(Y | X) \propto P(Y) \prod_{i=1}^{n} P(x_i | Y)
```

## 3. Types of Naïve Bayes Classifiers

### A. Gaussian Naïve Bayes (For Continuous Features)
When features follow a **normal distribution**, we use the Gaussian formula:

```math
P(x_i | Y) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x_i - \mu)^2}{2 \sigma^2}}
```

Where:
- `μ` = Mean of feature `x_i` for class `Y`
- `σ` = Standard deviation of feature `x_i` for class `Y`

### B. Multinomial Naïve Bayes (For Categorical Features)
Used when features represent frequency counts (e.g., word occurrences in text classification):

```math
P(x_i | Y) = \frac{count(x_i, Y) + 1}{\sum_{j} count(x_j, Y) + |V|}
```

Where:
- `count(x_i, Y)`: Number of times feature `x_i` appears in class `Y`
- `|V|`: Total number of unique features (Vocabulary size)
- **Laplace Smoothing** (+1) prevents zero probabilities

### C. Bernoulli Naïve Bayes (For Binary Features)
Used for binary features (0 or 1), such as presence/absence of a word in text classification:

```math
P(x_i | Y) =
\begin{cases}
P_i & \text{if } x_i = 1 \\
1 - P_i & \text{if } x_i = 0
\end{cases}
```

Where `P_i` is the probability of `x_i = 1` in class `Y`.

## 4. Class Prediction
For a given test feature vector `X = (x_1, x_2, ..., x_n)`, compute:

```math
P(Y_k | X) = P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
```

The predicted class `Y_pred` is:

```math
Y_{pred} = \arg\max_{Y_k} P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
```

## Summary
- **Bayes’ Theorem** is applied to classification.
- **Naïve Bayes Assumption** considers independent features.
- **Gaussian Naïve Bayes** handles continuous data using a normal distribution.
- **Multinomial Naïve Bayes** is used for text classification (word frequency).
- **Bernoulli Naïve Bayes** is used for binary features (word presence/absence).
- **Prediction** is made by selecting the class with the highest probability.

This method is computationally efficient and widely used in spam detection, text classification, and recommendation systems.

# Naïve Bayes Classifier with Hand Calculation

## Step 1: Given Raw Dataset
Consider a simple dataset with two features (Height and Weight) and a class label (Sport Type).

| Person | Height (cm) | Weight (kg) | Sport Type  |
|--------|------------|------------|-------------|
| A      | 180        | 80         | Basketball  |
| B      | 175        | 75         | Basketball  |
| C      | 190        | 85         | Basketball  |
| D      | 160        | 55         | Tennis      |
| E      | 165        | 60         | Tennis      |
| F      | 155        | 50         | Tennis      |

**Goal:** Predict the sport type for a new person with:  
- Height = 170 cm  
- Weight = 65 kg  

## Step 2: Calculate Prior Probabilities
\[
P(Basketball) = \frac{3}{6} = 0.5
\]
\[
P(Tennis) = \frac{3}{6} = 0.5
\]

## Step 3: Compute Mean and Standard Deviation
Using the **Gaussian Naïve Bayes** formula:
\[
P(x | Y) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
\]

### For Basketball Players
\[
\mu_{height} = \frac{180 + 175 + 190}{3} = 181.67
\]
\[
\sigma_{height} = \sqrt{\frac{(180-181.67)^2 + (175-181.67)^2 + (190-181.67)^2}{2}} = 7.57
\]
\[
\mu_{weight} = \frac{80 + 75 + 85}{3} = 80
\]
\[
\sigma_{weight} = \sqrt{\frac{(80-80)^2 + (75-80)^2 + (85-80)^2}{2}} = 5
\]

### For Tennis Players
\[
\mu_{height} = \frac{160 + 165 + 155}{3} = 160
\]
\[
\sigma_{height} = \sqrt{\frac{(160-160)^2 + (165-160)^2 + (155-160)^2}{2}} = 5
\]
\[
\mu_{weight} = \frac{55 + 60 + 50}{3} = 55
\]
\[
\sigma_{weight} = \sqrt{\frac{(55-55)^2 + (60-55)^2 + (50-55)^2}{2}} = 5
\]

## Step 4: Compute Likelihood Probabilities
For new person with **Height = 170 cm, Weight = 65 kg**, using the Gaussian formula:

### Basketball Probability
\[
P(170 | Basketball) = \frac{1}{\sqrt{2\pi (7.57)^2}} e^{-\frac{(170 - 181.67)^2}{2 (7.57)^2}} = 0.0213
\]
\[
P(65 | Basketball) = \frac{1}{\sqrt{2\pi (5)^2}} e^{-\frac{(65 - 80)^2}{2 (5)^2}} = 0.0007
\]

### Tennis Probability
\[
P(170 | Tennis) = \frac{1}{\sqrt{2\pi (5)^2}} e^{-\frac{(170 - 160)^2}{2 (5)^2}} = 0.0024
\]
\[
P(65 | Tennis) = \frac{1}{\sqrt{2\pi (5)^2}} e^{-\frac{(65 - 55)^2}{2 (5)^2}} = 0.0024
\]

## Step 5: Compute Posterior Probabilities
\[
P(Basketball | X) = P(170 | Basketball) P(65 | Basketball) P(Basketball)
\]
\[
= (0.0213) (0.0007) (0.5) = 7.46 \times 10^{-6}
\]

\[
P(Tennis | X) = P(170 | Tennis) P(65 | Tennis) P(Tennis)
\]
\[
= (0.0024) (0.0024) (0.5) = 2.88 \times 10^{-6}
\]

## Step 6: Prediction
Since \( P(Basketball | X) > P(Tennis | X) \), the predicted sport type is:

## Conclusion
- Bayes’ Theorem was applied step by step.
- Gaussian Probability Distribution was used for numerical features.
- The result shows that the new person is most likely a Basketball player.
