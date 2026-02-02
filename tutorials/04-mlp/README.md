# Backpropagation with a toy MLP
Before we move on to a full-featured toolbox, I wanted to provide you with something a bit simpler. I've written some (questionable) code in [mlp_regressor.py](mlp_regressor.py) to try to implement a multi-layer perceptron. I've also provided a starter notebook after throwing you to the wolves last week - you can download your copy [here](04-mlp.ipynb), or from [GitHub](https://github.com/MRU-COMP4630/w26/tree/main/tutorials/04-mlp/04-mlp.ipynb).

## Step 1: Load and preprocess data
We'll use a well-known and fairly clean dataset to try to predict [wine quality](https://archive.ics.uci.edu/dataset/186/wine+quality). You'll still need to encode (or ignore) the one categorical feature `color`, then split and normalize the inputs. You'll also need to `pip install ucimlrepo` to get the data-fetching module.

## Step 2: Build and train an MLP
The `MLPRegressor` class should be able to train a *small* multi-layer perceptron. You can use it like this:

```python
from mlp_regressor import MLPRegressor
mlp = MLPRegressor(X_train.shape[1])
mlp.add_layer(<number of neurons>, "activation function")
... repeat
print(mlp) # to see a summary of layers

loss = mlp.train(X_train, y_train, step_size, epochs)
plt.plot(loss)
```

It's very inefficient, so don't go too crazy with number of neurons. After training, you can predict by just running the forward pass:

```python
y_pred = mlp.forward(X_train)
```

There's also an example in the main block of `mlp_regressor.py`

## Step 3: Modify the MLP
Try to read through the forward and backward passes to understand how it works. It's entirely possible I've made a mistake somewhere, so don't hesitate to ask if something doesn't make sense.

To understand things in more detail, it can be helpful to try to modify it. Right now, the MLP only does whole-batch gradient descent. Can you modify it so that it does mini-batch or stochastic gradient descent?
