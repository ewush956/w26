# Assignment 2: Playing card classification
Due March 6, 2026 at 5 pm, with the usual weekend flexibility.

You may work in teams of 2 or 3. Click [here](https://classroom.github.com/a/HBYo1V77) to create your team on GitHub Classroom and clone the starter code. This can be the same team or different from assignment 1.

## Overview
The purpose of this assignment is to apply your theoretical knowledge of neural networks (particularly convolutional neural networks) to a real application. You will **build and train a model** from "scratch" (using PyTorch or some other framework, not really from scratch), and then see how much you can **reduce its size** while minimizing performance degradation. In addition to building and tweaking a neural network, this assignment serves as an introduction to:
- Working with a modern NN framework
- Preprocessing for image data
- Evaluating classification models

## Dataset
This time, I'm choosing the dataset: [Playing Cards](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification). This is quite a clean dataset with reasonable class balance. There are lots of implementations of classifiers using this dataset, and if you do look at someone else' work for ideas, make sure to **cite your sources** and **understand** what you are implementing.

> [!CAUTION]
> This dataset is very consistent compared to playing cards found in the wild. I will be evaluating on my own hand-curated dataset - you may want to augment yours with some extra card images as well.

## Deliverables
Your assignment should consist of the following:
1. Your notebook(s) and/or Python scripts where you did your experiments, with the final training run and evaluation rendered
2. A report describing your experiments and your final model decisions
3. Your final model classes in a Python module named `your_team_name.py`, alongside saved weights.  **Please make sure that your models load and run properly in Colab**. If additional packages are required, list them in your report document.

I would recommend working in parallel with your teammate(s) and commit your changes after each experiment. It's fine if you have multiple working notebooks, just indicate to me which is the final version.

### Your training code
Your training code (notebook or Python script) should:
- Load the training data and do some basic data exploration, like looking at samples, number of classes, class distribution, etc. The code in `starter.ipynb` provides some ideas for connecting Colab to Google Drive, defining the training dataset, and inspecting a few samples.
- Do any preprocessing you might want to do (at the very least, you'll probably want to rescale the images from unsigned ints in the range 0-255 to floats in the range 0 to 1)
  > Hint: Pytorch has some preprocessing layers that you can stick on the start of your model much like Scikit-learn's pipelines. Check out [Torchvision Transforms](https://docs.pytorch.org/vision/main/transforms.html) for more ideas.
- Define and train a model, keeping in mind the following:
    - The input layer must match the number of channels of your input. You do not need to define the batch size.
    - The output layer must have as many neurons as classes you are trying to predict.
    - Everything in between is a design choice that you can tweak!
- Iterate! I would suggest starting with a simple CNN feeding in to a fully connected output layer. I included my fairly random and not at all optimized model in the starter code.
- Train two models: one "best performance" version, where you try to get the highest **accuracy**, and one "size optimized" where you try to maintain reasonably good accuracy with the **fewest parameters**.
- Once you've trained your models, save the weights using `torch.save(model.state_dict())` as described [here](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). You may need to share these with me via Google Drive if they end up too big, but if not, you can just commit the binary weights file to your repo.

### Your production code
To load the weights for your model and run inference, PyTorch needs to know the class definition. There is a way of saving the whole thing at once, but it's fragile (much like pickling a Scikit-learn pipeline with custom functions).

After you're happy with both small and large models, copy the class definitions in a Python module named for your team name (e.g. `super_awesome_team.py`). Also copy over your transformation pipeline - I will be passing this to `ImageFolder` when I load the top secret evaluation set. You can either use the same one for small and large models, or two separate ones, or parameterize with image size, etc.

```python
from super_awesome_team import SmallModel, BigModel, img_transform
model = SmallModel()
model.load_state_dict(torch.load(path_to_small_weights, weights_only=True))
model.eval() # disables dropout and batch norm for inference
```

> [!TIP]
> Make sure to include any necessary preprocessing like rescaling in your transformation pipeline, but do not include image augmentation steps like `RandomHorizontalFlip`.

### Your report
In a separate document, summarize your experiments, models, observations, reflections, etc. I've provided a template with more details in the starter code (`report.md`), though you aren't limited to the markdown format.

## Marking Scheme
Each of the following components will be marked on a 4-point scale and weighted.

| Component                                               | Weight |
| ------------------------------------------------------- | ------ |
| Report: model development and experimentation           | 20%    |
| Report: reflections                                     | 20%    |
| Report: abstract and appendices                         | 20%    |
| Model: load and run on Colab                            | 20%    |
| Model: performance (highest accuracy model)             | 10%    |
| Model: performance / parameters ratio (size optimized)  | 10%    |

| Score | Description                                                            |
| ----- | ---------------------------------------------------------------------- |
| 4     | Excellent - thoughtful and creative without any errors or omissions    |
| 3     | Pretty good, but with minor errors or omissions                        |
| 2     | Mostly complete, but with major errors or omissions, lacking in detail |
| 1     | A minimal effort was made, incomplete or incorrect                     |
| 0     | No effort was made, or the submission is plagiarized                   |
