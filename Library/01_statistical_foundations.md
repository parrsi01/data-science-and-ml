# Statistical Foundations

---

> **Field** — Statistics and Probability
> **Scope** — Core statistical concepts used throughout
> data science, from descriptive measures to
> hypothesis testing and non-parametric methods

---

## Overview

Statistics is the backbone of data science.
Before you build models, generate charts,
or make predictions, you need to understand
how data behaves, how to summarize it,
and how to test whether the patterns you
see are real or just noise. This reference
covers the foundational statistical concepts
you will encounter in nearly every project.

---

## Definitions

### `Probability Distribution`

**Definition.**
A probability distribution describes how
the possible values of a variable are
spread out and how likely each value is.
Think of it as a map that tells you
"what values can happen and how often."

**Context.**
Distributions are everywhere in data science.
When you look at customer ages, sensor
readings, or exam scores, each dataset
follows some kind of distribution. Knowing
which distribution your data follows helps
you pick the right statistical tests and
build better models.

**Example.**
Three common distributions:

- **Normal (Gaussian):**
  The classic bell curve.
  Most values cluster around the mean,
  with fewer values far from center.
  Heights of adults follow this pattern.

- **Binomial:**
  Counts the number of successes in a
  fixed number of yes/no trials.
  Example: flip a coin 10 times,
  count the heads.

- **Poisson:**
  Counts how many times an event happens
  in a fixed interval of time or space.
  Example: how many emails you receive
  per hour.

```python
import numpy as np

# Normal distribution: 1000 samples,
# mean=50, std=10
normal_data = np.random.normal(50, 10, 1000)

# Binomial: 1000 experiments,
# 10 trials each, 50% success rate
binomial_data = np.random.binomial(10, 0.5, 1000)

# Poisson: 1000 samples,
# average rate = 3 events per interval
poisson_data = np.random.poisson(3, 1000)
```

---

### `Mean`

**Definition.**
The mean is the average of a set of numbers.
You calculate it by adding up all the values
and dividing by how many values there are.

**Context.**
The mean is the most commonly used measure
of central tendency in data science. It gives
you a single number that represents the
"center" of your data. However, it can be
misleading when your data has extreme values
(outliers) that pull the average up or down.

**Example.**
Given the dataset: 2, 4, 6, 8, 10

```
Mean = (2 + 4 + 6 + 8 + 10) / 5
Mean = 30 / 5
Mean = 6
```

```python
import numpy as np

data = [2, 4, 6, 8, 10]
mean_value = np.mean(data)
print(mean_value)  # 6.0
```

---

### `Median`

**Definition.**
The median is the middle value when you
sort your data from smallest to largest.
If there is an even number of values,
the median is the average of the two
middle values.

**Context.**
The median is more robust than the mean
when your data has outliers. For example,
if you are looking at household income
in a neighborhood and one billionaire
lives there, the median gives a much
more realistic picture of what "typical"
income looks like than the mean would.

**Example.**
Dataset: 3, 7, 9, 15, 100

Sorted: 3, 7, **9**, 15, 100

The median is 9 (the middle value).

Notice the mean would be 26.8, which is
pulled up by the outlier value of 100.

```python
import numpy as np

data = [3, 7, 9, 15, 100]
median_value = np.median(data)
print(median_value)  # 9.0
```

---

### `Standard Deviation`

**Definition.**
Standard deviation measures how spread out
the values in a dataset are from the mean.
A small standard deviation means values are
clustered close to the average. A large one
means values are spread far apart.

**Context.**
Standard deviation is one of the most
important measures in data science. It tells
you how "variable" your data is. When you
see error bars on a chart, they often
represent one standard deviation above and
below the mean. It is also used in feature
scaling, anomaly detection, and statistical
tests.

**Example.**
Dataset A: 48, 49, 50, 51, 52
(tightly clustered, small std dev)

Dataset B: 10, 30, 50, 70, 90
(widely spread, large std dev)

```python
import numpy as np

data_a = [48, 49, 50, 51, 52]
data_b = [10, 30, 50, 70, 90]

print(np.std(data_a))  # ~1.41
print(np.std(data_b))  # ~28.28
```

Both datasets have a mean of 50,
but their spreads are very different.

---

### `Variance`

**Definition.**
Variance is the average of the squared
differences between each data point and
the mean. It is the square of the standard
deviation.

**Context.**
Variance is closely related to standard
deviation but is harder to interpret
directly because its units are squared.
For example, if your data is in meters,
the variance is in meters-squared.
Despite this, variance appears in many
formulas in statistics and machine learning,
including ANOVA, PCA, and the bias-variance
tradeoff.

**Example.**
Dataset: 2, 4, 6

```
Mean = 4
Differences from mean: -2, 0, +2
Squared differences: 4, 0, 4
Variance = (4 + 0 + 4) / 3 = 2.67
Standard deviation = sqrt(2.67) = 1.63
```

```python
import numpy as np

data = [2, 4, 6]
print(np.var(data))   # 2.6667
print(np.std(data))   # 1.6330
```

---

### `Hypothesis Testing`

**Definition.**
Hypothesis testing is a structured way to
decide whether a pattern you see in data is
real or could have happened by chance.
You start with a "null hypothesis" (nothing
interesting is happening) and use data to
decide whether to reject it.

**Context.**
Hypothesis testing is how data scientists
make evidence-based decisions. For example,
if you run an A/B test on a website and
version B gets more clicks, you need
hypothesis testing to determine whether the
difference is meaningful or just random luck.
It is used in clinical trials, marketing,
manufacturing quality control, and every
field that relies on data.

**Example.**
Scenario: you think a new drug lowers blood
pressure. You compare a treatment group to
a control group.

- **Null hypothesis (H0):**
  The drug has no effect.
- **Alternative hypothesis (H1):**
  The drug lowers blood pressure.

You collect data, run a statistical test,
and get a p-value. If the p-value is small
enough (usually below 0.05), you reject
the null hypothesis and conclude the drug
likely has an effect.

```python
from scipy import stats

control = [130, 128, 135, 132, 131]
treatment = [122, 119, 125, 121, 120]

t_stat, p_value = stats.ttest_ind(
    control, treatment
)
print(f"p-value: {p_value:.4f}")
# If p < 0.05, reject null hypothesis
```

---

### `p-value`

**Definition.**
The p-value is the probability of seeing
results as extreme as (or more extreme than)
what you observed, assuming the null
hypothesis is true. A small p-value means
your result is unlikely to be due to chance.

**Context.**
The p-value is the most commonly reported
number in hypothesis testing. A p-value
below 0.05 is the traditional threshold for
"statistical significance," but this cutoff
is a convention, not a law. A p-value does
NOT tell you how large or important an
effect is. It only tells you how surprising
the data would be if nothing were happening.

**Example.**
You test whether a coin is fair by flipping
it 100 times and getting 65 heads.

- If the coin were fair, getting 65 or more
  heads is very unlikely.
- The p-value might be 0.002.
- Since 0.002 < 0.05, you reject the null
  hypothesis that the coin is fair.

```python
from scipy import stats

# Binomial test: 65 heads in 100 flips,
# testing if probability = 0.5
result = stats.binomtest(65, 100, 0.5)
print(f"p-value: {result.pvalue:.4f}")
```

---

### `Statistical Significance`

**Definition.**
A result is statistically significant when
the p-value falls below a pre-chosen
threshold (usually 0.05). This means the
observed result is unlikely to have occurred
by random chance alone.

**Context.**
Statistical significance is the standard
decision rule in science and data analysis.
However, "significant" does not mean
"important." A tiny, meaningless difference
can be statistically significant if you have
a very large sample. Always pair significance
with effect size (how big the difference is)
to understand whether the result actually
matters in practice.

**Example.**
- p-value = 0.03 with threshold 0.05:
  **Significant.** You reject H0.
- p-value = 0.12 with threshold 0.05:
  **Not significant.** You fail to reject H0.

Important: "fail to reject" is not the same
as "proved false." It means you did not find
strong enough evidence.

---

### `Confidence Interval`

**Definition.**
A confidence interval is a range of values
that is likely to contain the true population
parameter. A 95% confidence interval means
that if you repeated the experiment many
times, about 95% of the intervals you
calculate would contain the true value.

**Context.**
Confidence intervals are more informative
than point estimates because they show how
much uncertainty exists. When you see a
headline like "the average response time
is 2.5 seconds," the confidence interval
might be [2.1, 2.9], which tells you the
true value could be anywhere in that range.
Wider intervals mean more uncertainty.

**Example.**
You measure the heights of 100 people and
get a mean of 170 cm with a 95% confidence
interval of [168.2, 171.8].

This means you are 95% confident the true
average height of the population is between
168.2 and 171.8 cm.

```python
import numpy as np
from scipy import stats

data = np.random.normal(170, 5, 100)
confidence = 0.95
n = len(data)
mean = np.mean(data)
se = stats.sem(data)
interval = stats.t.interval(
    confidence, df=n-1, loc=mean, scale=se
)
print(f"Mean: {mean:.1f}")
print(f"95% CI: [{interval[0]:.1f}, "
      f"{interval[1]:.1f}]")
```

---

### `Correlation`

**Definition.**
Correlation measures the strength and
direction of the relationship between two
variables. It ranges from -1 (perfect
negative relationship) to +1 (perfect
positive relationship), with 0 meaning
no linear relationship.

**Context.**
Correlation is one of the first things
data scientists check when exploring data.
It helps you find which variables move
together. However, correlation does NOT
imply causation. Ice cream sales and
drowning deaths are correlated (both go up
in summer), but ice cream does not cause
drowning. The most common measure is the
Pearson correlation coefficient.

**Example.**
- Correlation = +0.95: strong positive
  (as X goes up, Y goes up)
- Correlation = -0.80: strong negative
  (as X goes up, Y goes down)
- Correlation = +0.05: essentially none

```python
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

r = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {r:.3f}")
# Output: ~0.993 (strong positive)
```

---

### `Monte Carlo Simulation`

**Definition.**
A Monte Carlo simulation is a technique
that uses random sampling to estimate
outcomes that are difficult or impossible
to calculate analytically. You run the
simulation thousands or millions of times
to build a picture of what might happen.

**Context.**
Monte Carlo methods are used in finance
(portfolio risk), physics (particle
interactions), engineering (reliability
testing), and data science (uncertainty
estimation). Whenever a problem has too
many variables or too much complexity
for a closed-form solution, Monte Carlo
simulation is a practical alternative.

**Example.**
Estimating the probability of a portfolio
losing more than 10% in a month:

```python
import numpy as np

np.random.seed(42)
n_simulations = 100_000
daily_returns = np.random.normal(
    0.001, 0.02, (n_simulations, 21)
)
monthly_returns = np.prod(
    1 + daily_returns, axis=1
) - 1

prob_loss = np.mean(monthly_returns < -0.10)
print(f"Probability of >10% loss: "
      f"{prob_loss:.2%}")
```

Each simulation generates 21 random
daily returns and combines them into a
monthly return. By repeating 100,000 times,
you get a reliable estimate of the risk.

---

### `Sampling Bias`

**Definition.**
Sampling bias occurs when the sample you
collect does not accurately represent the
population you care about. Certain groups
or values are over-represented or
under-represented.

**Context.**
Sampling bias is one of the most common
mistakes in data science. If your training
data is biased, your model will be biased.
For example, if you train a loan approval
model on data that only includes approved
loans, it will never learn from the
characteristics of denied applicants.
Always ask: "Does my sample represent the
full population?"

**Example.**
Suppose you survey people about smartphone
preferences by posting a poll on Twitter.

- **Bias:** your sample only includes
  Twitter users, who tend to be younger
  and more tech-savvy than the general
  population.
- **Result:** your survey overestimates
  the popularity of newer, high-tech phones.
- **Fix:** use random sampling from a
  broader population (e.g., phone surveys
  or mail surveys).

---

### `Central Limit Theorem`

**Definition.**
The Central Limit Theorem (CLT) states that
when you take many random samples from any
population and compute their means, those
sample means will form an approximately
normal distribution, regardless of the
original population's shape. This holds as
long as the sample size is large enough
(usually 30 or more).

**Context.**
The CLT is one of the most important
theorems in statistics. It is the reason
we can use normal-distribution-based methods
(like z-tests and confidence intervals)
even when the underlying data is not
normally distributed. It justifies many
of the statistical techniques used in
data science.

**Example.**
Rolling a single die gives a uniform
distribution (each number 1-6 is equally
likely). But if you roll 50 dice and
take the average, then repeat this
thousands of times, the distribution of
those averages will look like a bell curve.

```python
import numpy as np

np.random.seed(42)
sample_means = []
for _ in range(10_000):
    rolls = np.random.randint(1, 7, size=50)
    sample_means.append(np.mean(rolls))

# sample_means will be approximately
# normally distributed around 3.5
print(f"Mean of means: "
      f"{np.mean(sample_means):.2f}")
print(f"Std of means: "
      f"{np.std(sample_means):.2f}")
```

---

### `Standard Error`

**Definition.**
The standard error (SE) is the standard
deviation of the sampling distribution
of a statistic (most commonly the mean).
It measures how much your sample mean is
likely to vary from sample to sample.

**Context.**
Standard error is closely related to the
Central Limit Theorem. A smaller standard
error means your sample mean is a more
precise estimate of the population mean.
Standard error decreases as your sample
size increases, which is why larger samples
give more reliable results. It is used in
constructing confidence intervals and
performing hypothesis tests.

**Example.**
If your data has a standard deviation of 10
and your sample size is 100:

```
SE = standard deviation / sqrt(sample size)
SE = 10 / sqrt(100)
SE = 10 / 10
SE = 1.0
```

```python
from scipy import stats
import numpy as np

data = np.random.normal(50, 10, 100)
se = stats.sem(data)
print(f"Standard error: {se:.2f}")
# Approximately 1.0
```

---

### `Wilcoxon Test`

**Definition.**
The Wilcoxon signed-rank test is a
non-parametric test used to compare two
related samples (paired observations).
It does not assume the data is normally
distributed, making it useful when that
assumption is violated.

**Context.**
Use the Wilcoxon test when you have paired
data (before/after measurements on the same
subjects) and your data is not normally
distributed or you have a small sample size.
It is the non-parametric alternative to the
paired t-test. Common in medical studies,
psychology experiments, and A/B tests with
matched pairs.

**Example.**
You measure task completion time before
and after a UI redesign for the same users:

```python
from scipy import stats

before = [12, 15, 14, 10, 13, 16, 11]
after  = [10, 13, 12,  9, 11, 14, 10]

stat, p_value = stats.wilcoxon(
    before, after
)
print(f"Wilcoxon statistic: {stat}")
print(f"p-value: {p_value:.4f}")
# If p < 0.05, the redesign had a
# significant effect on task time
```

---

### `Mann-Whitney U Test`

**Definition.**
The Mann-Whitney U test is a non-parametric
test used to compare two independent groups
to determine whether they come from the
same distribution. It does not require
normal data.

**Context.**
Use Mann-Whitney when you want to compare
two independent groups (not paired) and
your data is not normally distributed. It
is the non-parametric alternative to the
independent samples t-test. Common examples
include comparing satisfaction scores
between two product versions or comparing
test scores between two schools.

**Example.**
Comparing customer satisfaction scores
between two stores:

```python
from scipy import stats

store_a = [7, 8, 6, 9, 7, 8, 6]
store_b = [5, 6, 4, 7, 5, 6, 5]

stat, p_value = stats.mannwhitneyu(
    store_a, store_b,
    alternative='two-sided'
)
print(f"U statistic: {stat}")
print(f"p-value: {p_value:.4f}")
# If p < 0.05, the stores have
# significantly different scores
```

---

### `t-test (paired)`

**Definition.**
A paired t-test compares the means of two
related groups to determine if there is a
statistically significant difference between
them. "Paired" means each observation in one
group has a corresponding observation in the
other group (e.g., the same person measured
before and after a treatment).

**Context.**
The paired t-test is one of the most common
statistical tests in data science. Use it
when you have before/after data on the same
subjects and the differences are approximately
normally distributed. If the normality
assumption does not hold, use the Wilcoxon
signed-rank test instead.

**Example.**
Blood pressure before and after medication
for the same patients:

```python
from scipy import stats

before = [140, 135, 150, 145, 138]
after  = [130, 128, 140, 138, 132]

t_stat, p_value = stats.ttest_rel(
    before, after
)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
# ttest_rel is for paired (related) data
# ttest_ind is for independent data
```

---

### `Effect Size`

**Definition.**
Effect size is a measure of how large or
meaningful a difference or relationship is,
independent of sample size. Common measures
include Cohen's d (for group differences)
and Pearson's r (for correlations).

**Context.**
Effect size answers the question "how big
is the difference?" while the p-value only
answers "is there a difference?" A result
can be statistically significant but have a
tiny effect size, meaning it is practically
irrelevant. Always report effect size
alongside p-values. This is especially
important with large datasets where even
trivial differences become significant.

**Example.**
Cohen's d guidelines:

- 0.2 = small effect
- 0.5 = medium effect
- 0.8 = large effect

```python
import numpy as np

group_a = [85, 90, 88, 92, 87]
group_b = [78, 82, 80, 84, 79]

mean_diff = np.mean(group_a) - np.mean(group_b)
pooled_std = np.sqrt(
    (np.std(group_a, ddof=1)**2 +
     np.std(group_b, ddof=1)**2) / 2
)
cohens_d = mean_diff / pooled_std
print(f"Cohen's d: {cohens_d:.2f}")
# d > 0.8 = large effect
```

---

## See Also

- [Python and Numerical Computing](./02_python_and_numerical_computing.md)
- [Machine Learning Fundamentals](./05_machine_learning_fundamentals.md)
- [Model Evaluation and Monitoring](./07_model_evaluation_and_monitoring.md)

---

> **Author** — Simon Parris | Data Science Reference Library
