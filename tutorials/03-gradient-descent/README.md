# Gradient Descent for Polynomial Regression
There's some fake data in the file [data.csv](data.csv), with a single feature `x` and a true value `y`. Your task is to:
1. Load the data and look at it
2. Split it into training, validation, and test sets
3. Create your design matrix
4. Implement gradient descent to find the best fit polynomial
5. Evaluate your model's performance and experiment with different hyperparameters

It's up to you to decide what degree polynomial to fit the data, and you can also play around with stochastic gradient descent, mini-batch, hyperparameters, etc.

> [!IMPORTANT]
> Do this without the use of scikit learn or other libraries aside from `numpy` and `matplotlib`!

## Step 0: Import libraries and seed your random number generator
It's usually a good idea to start with a consistent random number seed to ensure reproducibility.
```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed="integer_of_your_choice")
```

## Step 1: Load the data and look at it

```python
x, y = np.loadtxt("data.csv", delimiter=",", skiprows=1, unpack=True)
#TODO: visualize
```

## Step 2: Split the data
Weird numpy quirk: by default, a 1D array has a shape of `(n,)`, but to behave as a proper vector, we need to convert it to be `(n, 1)`. An easy way to do this is to pass `np.newaxis` as the second index when sampling your `y` data, e.g.:

```python
n = len(y)
train_ids = rng.choice()
x_train, y_train = x[train_ids,], y[train_ids, np.newaxis]
```

Don't worry about the x values for now, as we'll be matrixifying them shortly anyway.

## Step 3: Create your **design matrix** $X$.

For the example given in class, the design matrix was simply a column of 1s concatenated with the feature vector, i.e.:

$$X = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_m \end{bmatrix}$$

For this exercise, you probably want to fit a higher degree polynomial, so the design matrix will be something like:

$$X = \begin{bmatrix} 1 & x_1 & x_1^2 & \ldots & x_1^d \\ 1 & x_2 & x_2^2 & \ldots & x_2^d \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ 1 & x_m & x_m^2 & \ldots & x_m^d \end{bmatrix}$$

where $d$ is the degree of the polynomial you want to fit. Try multiple degrees and see what gives the best results.

> A note on scaling: the range of x values in this example is fairly small, but if you choose a high degree polynomial you will still end up with fairly different scales for your "features". Consider normalizing each column of the design matrix (other than the first column accounting for the bias term), remembering to calculate your scaling parameters on the training data and apply them to the validation/test data.

Since you'll be doing this twice (train/test), you might want to define a function to create the design matrix given a vector x and a degree d.

## Step 4: Implement gradient descent
This has a number of sub components. First you'll need to define your gradient function. For mean squared error, the gradient can be calculated as:

$$\nabla_{\theta} MSE = \frac{2}{m}X^T(X\mathbf{\theta} - \mathbf{y})$$

where $X$ is your design matrix, $\mathbf{\theta}$ is the current parameter vector, and $\mathbf{y}$ is the true target value.

It'll also be useful to define the actual mean squared error to evaluate your model:

$$MSE = \frac{1}{m}(\mathbf{X} \mathbf{\theta} - \mathbf{y})^T (\mathbf{X} \mathbf{\theta} - \mathbf{y})$$

Now you can define your hyperparameters and run your gradient descent. For batch gradient descent, you'll need to define:
- learning rate $\eta$ (usually in the range of $10^{-5}$ to $10^{-2}$)
- stopping criterion (can just be a fixed number of iterations)

The general algorithm for gradient descent is:
1. Start with a random $\mathbf{\theta}$
2. Calculate the gradient $\nabla_{\mathbf{\theta}}$ for the current $\mathbf{\theta}$
3. Update $\mathbf{\theta}$ as $\mathbf{\theta} = \mathbf{\theta} - \eta \nabla_{\mathbf{\theta}}$
4. Repeat 2-4 until some stopping criterion is met

You could also try mini-batch or stochastic gradient descent by adding an outer epoch loop if you want to get fancy.

## Step 5: Evaluate your model's performance and experiment
Now that you've computed a final estimate of $\mathbf{\theta}$, apply it to your test set to see how well your model performs, perhaps by plotting the data as well as the best fit curve. If it doesn't look good, try changing various hyperparameters, like $\eta$, number of iterations, and degree of polynomial. If you didn't rescale your design matrix earlier, try it now!

> Technically we should have done a 3-way train/validate/test split, but I kept it as just train/test to keep things manageable.
