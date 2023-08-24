# Statistics in ML & DL Cheat Sheet

## Table of Contents
1. [Basics (Statistics From Scratch)](#basics-from-scratch)
2. [Advanced statistics](#advanced)
3. [Linear Algebra](#linear-algebra)
4. [Calculus](#calculus)
5. [Numerical Optimization](#numerical-optimization)
6. [Measurement Theory](#measurement-theory)

---

## Basics (From Scratch)

### Descriptive Statistics:
Descriptive statistics provide a summary of the main aspects of the data, providing a snapshot of its main characteristics.

- **Mean:** The average of all data points. It's the sum of all data points divided by the number of data points.
- **Median:** The middle value in sorted data. If there's an even number of observations, it's the average of the two middle numbers.
- **Mode:** The value that appears most frequently in a data set. A dataset can have one mode, more than one mode, or no mode at all.
- **Variance:** Measures how far each number in a set is from the mean and is calculated by taking the average of the squared differences from the mean.
- **Standard Deviation:** Represents the average distance between each data point and the mean. It's the square root of variance.
- **Skewness:** Measures the asymmetry of the probability distribution of a random variable about its mean. Positive skewness indicates a distribution with an asymmetric tail extending toward more positive values.
- **Kurtosis:** Describes the tails and sharpness of a distribution. Higher kurtosis means more of the variance is the result of infrequent extreme deviations, as opposed to frequent modestly sized deviations.
- **Range:** The difference between the highest and lowest values in a dataset.
- **Interquartile Range (IQR):** Represents the middle 50% of the data. It's the difference between the third quartile (Q3) and the first quartile (Q1).

### Probability Distributions:
Probability distributions describe how the values of a random variable are distributed.

- **Normal Distribution:** A symmetrical, bell-shaped curve where most observations cluster around the central peak and the probabilities for values further away from the mean taper off equally in both directions.
- **Binomial Distribution:** Represents the number of successes in a fixed number of binary experiments. It describes the outcome of binary scenarios, e.g., toss of a coin, win or loss, etc.
- **Poisson Distribution:** Represents the number of events in fixed intervals of time or space. It's used for counting the number of times an event occurs within a given time or space.
- **Exponential Distribution:** Describes the time between events in a Poisson process. It's often used to model the time elapsed between events.
- **Uniform Distribution:** All outcomes are equally likely, and every variable has the same probability that it will be the outcome.
- **Gamma Distribution:** A two-parameter family of continuous probability distributions. It generalizes the exponential distribution.
- **Beta Distribution:** Models random variables limited to intervals of finite length in the range from 0 to 1.

### Hypothesis Testing:
Hypothesis testing is a statistical method used to make decisions or inferences about population parameters based on a sample.

- **Null Hypothesis (H0):** The initial claim or status quo. It's a statement of no effect or no difference.
- **Alternative Hypothesis (H1 or Ha):** Represents what the researcher aims to prove. It's a statement indicating the presence of an effect or difference.
- **Type I Error (α):** The probability of rejecting the null hypothesis when it's actually true. Commonly referred to as a "false positive."
- **Type II Error (β):** The probability of not rejecting the null hypothesis when the alternative hypothesis is true. Known as a "false negative."
- **Chi-Squared Test:** Used for testing relationships between categorical variables. It compares the observed frequencies to the frequencies we would expect if there were no relationship between the variables.
- **F-Test:** Compares the variances of two populations. It's used to test if two population variances are equal.
- **Central Limit Theorem:** States that the distribution of sample means approaches a normal distribution as the sample size gets larger, regardless of the shape of the population distribution.

### Correlation & Sampling:
Correlation measures the strength and direction of a linear relationship between two variables.

- **Pearson Correlation Coefficient (r):** Measures the linear relationship between two variables. It ranges from -1 to 1, with -1 indicating a perfect negative linear relationship and 1 indicating a perfect positive linear relationship.
- **Spearman Rank Correlation:** A non-parametric test that measures the strength and direction of the relationship between two variables. It assesses how well the relationship between two variables can be described using a monotonic function.
- **Random Sampling:** A subset of individuals chosen from a larger set where each individual has an equal chance of being selected.
- **Stratified Sampling:** Divides the population into subgroups or strata, and random samples are taken from each stratum.

### Estimation & Experimental Design:
Estimation involves approximating a population parameter based on a sample from the population.

- **Point Estimation:** Provides a single value as an estimate of a parameter.
- **Interval Estimation:** Provides a range of values as an estimate of a parameter. This range is called a confidence interval.
- **Factorial Experiments:** Experiments that study the effects of multiple factors simultaneously.
- **Cross-Over Design:** An experimental design in which subjects receive different treatments during the different periods of the experiment.

---

## Advanced

### Inferential & Multivariate Analysis:
Inferential statistics allow you to infer or deduce population parameters from sample statistics.

- **Z-Score:** Represents how many standard deviations a data point is from the mean. It's used to compare scores from different distributions.
- **T-Test:** A hypothesis test used to compare the means of two groups. It assumes that the populations the samples come from are normally distributed.
- **Cohen's d:** A measure of effect size that indicates the standardized difference between two means.
- **Effect Size:** Represents the magnitude of a relationship between two or more variables. It describes the strength of the relationship between variables.
- **Canonical Correlation:** Measures the association between two sets of variables. It's used to find the relationship between two sets of variables if each set is used to define a new variable.
- **Discriminant Analysis:** A classification technique. It determines which variables discriminate between two or more naturally occurring groups.

### Bayesian Statistics & Time Series Analysis:
Bayesian statistics is a statistical method that applies probability to statistical problems, involving epistemological uncertainties.

- **Bayes' Theorem:** Provides a way to revise existing predictions or theories (update probabilities) given new evidence. It relates current evidence to prior beliefs.
- **Prior Distribution:** Represents what we know about a variable before considering the current observed evidence.
- **Posterior Distribution:** Represents what we know about a variable after considering the current observed evidence.
- **Autoregression (AR):** A time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step.
- **Moving Average (MA):** A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
- **ARIMA (AutoRegressive Integrated Moving Average):** A class of models that combines AR and MA models as well as a differencing pre-processing step of the sequence to make the sequence stationary.

### Non-Parametric & Robust Statistics:
Non-parametric statistics don't assume that the data follows a particular distribution.

- **Mann-Whitney U Test:** A non-parametric test used to compare two sample means that come from the same population and used to test if two population means are equal.
- **Wilcoxon Signed-Rank Test:** A non-parametric test that compares two paired groups.
- **Kruskal-Wallis H Test:** A rank-based non-parametric test that can be used to determine if there are statistically significant differences between two or more groups of an independent variable on a continuous or ordinal dependent variable.
- **Huber's M-Estimator:** A method used in robust regression that is less sensitive to outliers in data than the least squares method.

### Causal Inference & Survival Analysis:
Causal inference is the process of drawing a conclusion about a causal connection based on the conditions of the occurrence of an effect.

- **Randomized Controlled Trials (RCTs):** An experimental setup where participants are randomly assigned to an experimental group or a control group.
- **Propensity Score Matching:** A statistical matching technique that attempts to estimate the effect of a treatment by accounting for the covariates that predict receiving the treatment.
- **Kaplan-Meier Estimator:** A non-parametric statistic used to estimate the survival function from lifetime data.
- **Cox Proportional-Hazards Model:** A regression model commonly used in medical research for investigating the association between the survival time of patients and one or more predictor variables.

---

## Linear Algebra

### Basics:
Linear algebra is a branch of mathematics concerning linear equations, linear functions, and their representations through matrices and vector spaces.

- **Vector:** A quantity having direction and magnitude. It can be represented as an array of numbers.
- **Matrix:** A rectangular array of numbers, symbols, or expressions, arranged in rows and columns.
- **Transpose:** A matrix derived by interchanging rows and columns.
- **Determinant:** A scalar value derived from a square matrix and tells if the matrix has an inverse.
- **Inverse:** A matrix, when multiplied by the original matrix, results in the identity matrix.
- **Eigenvalues and Eigenvectors:** Scalar and vector values, respectively, that satisfy the equation `A*v = lambda*v` where A is a matrix, v is the eigenvector, and lambda is the eigenvalue.
- **Dot Product:** A scalar product of two vectors, representing the cosine of the angle between them.
- **Cross Product:** A vector product of two vectors, resulting in a vector perpendicular to both.
- **Orthogonal:** Two vectors are orthogonal if their dot product is zero, meaning they are at right angles to each other.
- **Orthonormal:** A set of vectors that are both orthogonal and of unit length.

### Matrix Factorizations:
Matrix factorization is the breaking down of one matrix into a product of multiple matrices.

- **Row Echelon Form:** A form of a matrix where the leading entry of each row is to the right of the leading entry of the previous row. It's used to solve systems of linear equations.
- **Reduced Row Echelon Form:** A matrix in row echelon form with additional properties like leading entries are 1 and the only non-zero entries in their columns. It provides a unique solution to the system of equations.
- **Gaussian Elimination:** A method to solve systems of linear equations by transforming the system to an upper triangular matrix. It's a sequence of operations performed on the associated matrix of coefficients.
- **LU Decomposition:** Decomposing a matrix into a product of a lower triangular matrix (L) and an upper triangular matrix (U). It's used to simplify the solving of systems of linear equations, such as finding the determinants and inverses.

---

## Calculus

### Basics:
Calculus is a branch of mathematics that studies continuous change, via derivatives and integrals.

- **Limit:** The value a function approaches as the input approaches a certain value. It's foundational to understanding the concepts of derivatives and integrals.
- **Derivative:** Measures how a function changes as its input changes. It represents the slope of the tangent line to the function at any point.
- **Integral:** Represents the area under a curve. It can be thought of as the "opposite" of differentiation.
- **Partial Derivative:** Derivative of a function with respect to one of its variables, treating the other variables as constants. It's used in multivariable calculus.
- **Gradient:** A vector containing all the partial derivatives of a function. It points in the direction of the greatest rate of increase of the function.
- **Chain Rule:** A formula to compute the derivative of a composite function. It's used when differentiating a function that has another function inside it.
- **Product Rule:** A formula to compute the derivative of the product of two functions.
- **Quotient Rule:** A formula to compute the derivative of the quotient of two functions.

### Techniques of Integration:
Integration techniques are methods used to find antiderivatives and integrals of functions.

- **Integration by Parts:** A method based on the product rule for differentiation. It's used to integrate products of functions.
- **Integration by Substitution:** Changing the variable of integration to simplify the integral. It's similar to the chain rule for differentiation.
- **Partial Fraction Decomposition:** Expressing a rational function as a sum of simpler fractions to simplify the integral. It's used for integrating rational functions.

### Multivariable Calculus:
Multivariable calculus involves calculus of more than one variable.

- **Double Integral:** An integral over a region in the plane. It gives the volume under the surface defined by a function of two variables.
- **Triple Integral:** An integral over a region in space. It gives the volume of a region in space defined by a function of three variables.
- **Line Integral:** An integral over a curve in the plane or space. It's used to find the work done by a force field in moving a particle along a curve.
- **Surface Integral:** An integral over a surface in space. It's used to find the flux of a vector field across a surface.
- **Divergence:** A scalar measure of a vector field's source or sink at a given point. It represents the rate at which "density" exits a given region of space.
- **Curl:** A vector measure of a vector field's rotation at a given point. It measures the rotation or circulation of a vector field.
- **Stokes' Theorem:** Relates a surface integral of a vector field to a line integral around the boundary of the surface. It's a fundamental theorem in vector calculus.
- **Green's Theorem:** Relates a line integral around a simple closed curve C to a double integral over the plane region bounded by C. It's a special case of Stokes' theorem.
- **Divergence Theorem:** Relates a triple integral over a volume bounded by a closed surface S to a surface integral over S. It's used to transform volume integrals to surface integrals.

---

## Numerical Optimization

### Basics:
Numerical optimization involves finding the best solution or approximation to a problem using numerical methods.

- **Gradient Descent:** An iterative optimization algorithm used to find the minimum of a function. It works by taking steps proportional to the negative of the gradient of the function at the current point.
- **Newton's Method:** An iterative method used to find successively better approximations to the roots (or zeros) of a real-valued function.
- **Conjugate Gradient:** An algorithm for the numerical solution of particular systems of linear equations. It's often used when the system has a large number of equations.
- **Lagrange Multipliers:** A method for finding the local maxima and minima of a function subject to equality constraints.
- **Simplex Algorithm:** A popular algorithm for numerical solution of linear programming problems.

### Constraints & Regularization:
Constraints restrict the feasible solutions in optimization, while regularization adds penalties to prevent overfitting.

- **Bound Constraints:** Limit the values of the decision variables to lie in a specified range.
- **Linear Constraints:** Require linear functions of the decision variables to be less than, equal to, or greater than a constant.
- **Nonlinear Constraints:** Involve nonlinear functions of the decision variables.
- **L1 Regularization (Lasso):** Adds a penalty equal to the absolute value of the magnitude of coefficients. It can lead to some coefficients being exactly zero.
- **L2 Regularization (Ridge):** Adds a penalty equal to the square of the magnitude of coefficients. It prevents overfitting by penalizing large coefficients.

---

## Measurement Theory

### Basics:
Measurement theory deals with the assignment of numbers to objects or events according to specific rules.

- **Nominal Scale:** A scale of measurement where numbers serve only as labels and do not have any quantitative significance.
- **Ordinal Scale:** A scale of measurement that allows for rank order (1st, 2nd, 3rd, etc.) by which data can be sorted, but differences between data are not meaningful.
- **Interval Scale:** A scale of measurement where the difference between two values is meaningful. It doesn't have a true zero point.
- **Ratio Scale:** A scale of measurement that has a true zero point. It's the most informative scale.
- **Reliability:** The consistency or repeatability of a measure. If the same test is repeated, it should produce the same results.
- **Validity:** The extent to which a test measures what it claims to measure. It ensures that the instrument is accurate.

### Scaling & Transformation:
Scaling and transformation involve changing the scale or values of a variable.

- **Z-Score Scaling:** Transforms data into a standard normal distribution with a mean of 0 and a standard deviation of 1.
- **Min-Max Scaling:** Transforms data to fit within a specified range, usually [0,1].
- **Log Transformation:** Used to transform skewed data to approximately conform to normality.
- **Box-Cox Transformation:** A family of power transformations that are used to stabilize variance and make the data more normal in distribution.

---

