# Naïve Bayes Theorem and Feature Formulas

## 1. Naïve Bayes Theorem
The Naïve Bayes classifier is a probabilistic model based on **Bayes’ theorem**, assuming independence among features.

$$
P(Y | X) = \frac{P(X | Y) P(Y)}{P(X)}
$$

Where:
- \( P(Y | X) \): Posterior probability (Probability of class \( Y \) given feature vector \( X \))
- \( P(X | Y) \): Likelihood (Probability of \( X \) given class \( Y \))
- \( P(Y) \): Prior probability (Probability of class \( Y \) before observing features)
- \( P(X) \): Evidence (Probability of \( X \) across all classes)

Since \( P(X) \) is constant for all classes:

$$
P(Y | X) \propto P(X | Y) P(Y)
$$

## 2. Naïve Bayes Assumption
The model assumes each feature is conditionally independent given the class label \( Y \):

$$
P(X | Y) = P(x_1, x_2, ..., x_n | Y) = P(x_1 | Y) P(x_2 | Y) ... P(x_n | Y)
$$

Thus, the simplified formula becomes:

$$
P(Y | X) \propto P(Y) \prod_{i=1}^{n} P(x_i | Y)
$$

## 3. Types of Naïve Bayes Classifiers

### A. Gaussian Naïve Bayes (For Continuous Features)
When features follow a **normal distribution**, we use the Gaussian formula:

$$
P(x_i | Y) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x_i - \mu)^2}{2 \sigma^2}}
$$

Where:
- \( \mu \) = Mean of feature \( x_i \) for class \( Y \)
- \( \sigma \) = Standard deviation of feature \( x_i \) for class \( Y \)

### B. Multinomial Naïve Bayes (For Categorical Features)
Used when features represent frequency counts (e.g., word occurrences in text classification):

$$
P(x_i | Y) = \frac{count(x_i, Y) + 1}{\sum_{j} count(x_j, Y) + |V|}
$$

Where:
- \( count(x_i, Y) \): Number of times feature \( x_i \) appears in class \( Y \)
- \( |V| \): Total number of unique features (Vocabulary size)
- **Laplace Smoothing** (+1) prevents zero probabilities

### C. Bernoulli Naïve Bayes (For Binary Features)
Used for binary features (0 or 1), such as presence/absence of a word in text classification:

$$
P(x_i | Y) =
\begin{cases}
P_i & \text{if } x_i = 1 \\
1 - P_i & \text{if } x_i = 0
\end{cases}
$$

Where \( P_i \) is the probability of \( x_i = 1 \) in class \( Y \).

## 4. Class Prediction
For a given test feature vector \( X = (x_1, x_2, ..., x_n) \), compute:

$$
P(Y_k | X) = P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
$$

The predicted class \( Y_{pred} \) is:

$$
Y_{pred} = \arg\max_{Y_k} P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
$$

## Summary
- **Bayes’ Theorem** is applied to classification.
- **Naïve Bayes Assumption** considers independent features.
- **Gaussian Naïve Bayes** handles continuous data using a normal distribution.
- **Multinomial Naïve Bayes** is used for text classification (word frequency).
- **Bernoulli Naïve Bayes** is used for binary features (word presence/absence).
- **Prediction** is made by selecting the class with the highest probability.

This method is computationally efficient and widely used in spam detection, text classification, and recommendation systems.

# Naïve Bayes: Real-Time Example (Spam Detection)

## Problem Statement
We want to classify an **email** as **spam or not spam** based on the presence of certain words.

## 1. Applying Bayes’ Theorem
Using Naïve Bayes, we calculate:

$$
P(Spam | Words) = \frac{P(Words | Spam) P(Spam)}{P(Words)}
$$

Since \( P(Words) \) is constant for both classes:

$$
P(Spam | Words) \propto P(Words | Spam) P(Spam)
$$

Similarly, for **Not Spam (Ham):**

$$
P(Ham | Words) \propto P(Words | Ham) P(Ham)
$$

## 2. Naïve Bayes Assumption
Each word is conditionally independent given the class:

$$
P(Words | Spam) = P(w_1 | Spam) P(w_2 | Spam) ... P(w_n | Spam)
$$

Thus, for prediction:

$$
P(Spam | Words) \propto P(Spam) \prod_{i=1}^{n} P(w_i | Spam)
$$

## 3. Example Dataset

| Email ID | Words in Email | Spam (Y/N) |
|----------|---------------|------------|
| 1        | "Win lottery now" | Yes |
| 2        | "Hello, how are you?" | No |
| 3        | "Win cash prize today" | Yes |
| 4        | "Meeting at 5 PM" | No |

### **Word Probabilities**
From training data:

$$
P(Spam) = \frac{2}{4} = 0.5, \quad P(Ham) = \frac{2}{4} = 0.5
$$

For word **"Win"**:

$$
P(Win | Spam) = \frac{2}{2} = 1, \quad P(Win | Ham) = \frac{0}{2} = 0
$$

For word **"Hello"**:

$$
P(Hello | Spam) = \frac{0}{2} = 0, \quad P(Hello | Ham) = \frac{1}{2} = 0.5
$$

Applying **Laplace Smoothing (+1 in numerator, +Total Vocabulary in denominator):**

$$
P(Win | Spam) = \frac{2+1}{2+4} = \frac{3}{6} = 0.5
$$

$$
P(Win | Ham) = \frac{0+1}{2+4} = \frac{1}{6} \approx 0.167
$$

## 4. Predicting a New Email:  
**Email: "Win cash now"**

We calculate:

$$
P(Spam | Email) \propto P(Spam) P(Win | Spam) P(Cash | Spam) P(Now | Spam)
$$

$$
P(Ham | Email) \propto P(Ham) P(Win | Ham) P(Cash | Ham) P(Now | Ham)
$$

If \( P(Spam | Email) > P(Ham | Email) \), classify as **Spam**, else **Not Spam**.

## 5. Conclusion
- **Naïve Bayes is useful for spam filtering.**
- **Each word contributes to the probability calculation.**
- **Laplace smoothing prevents zero probabilities.**

