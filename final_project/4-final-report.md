# 3. Final Report
This time, I'd like you to format according to the [NeurIPS 2015 style guide](https://neurips.cc/Conferences/2015/PaperInformation/StyleFiles). This was the last year that Word documents were accepted, so I decided to use this one instead of making you all use LaTeX (though of course, LaTeX is always recommended).

Instead of a maximum of 8 pages, I would like a **maximum of 4 pages** (a common "short paper" length for conferences). This includes figures, so make the most of your space! You can include additional figures in an appendix if you like, but the main content should be in the main body of the report.

The exact sections and titles are up to you, but the report should include the following:

## Abstract
This should be as short as possible (one paragraph) and should provide an "at a glance" summary of the task, dataset, and final results. Include a link to your code repository and/or model on Google Drive (if too big for GitHub).

## Introduction
Describe both the task and dataset, along with relevant citations and a brief discussion of prior work. Don't go too crazy down the research rabbit hole, but do try to find at least 2-3 papers that are relevant to your task.

If you modified your plans, include a brief description of why things changed (e.g. "Originally we planned to X, but the dataset turned out to have Y issues, so we switched to predicting Z"). Similarly, if you achieved your backup goal but not your original, briefly summarize why.

## Dataset and Preprocessing
Describe the dataset you used, including the number of samples, features, and classes (if applicable). Discuss any preprocessing steps you took, such as normalization, feature engineering, or handling missing values. Include figures and/or tables - whatever best conveys the information.

## Model Architecture and Training
Describe the model architecture you used and the training parameters. Describe any variations you tried but were not successful, and what contributed to the success of your final model.

## Discussion and Conclusion
Discuss your results, highlighting lessons learned from the process. What limitations did you encounter, and what further work would you do if you were to continue this project?

## References
The usual reference list, formatted according to the NeurIPS guidelines.

For each of the above sections, consider using figures and tables instead of dense paragraphs listing parameters or results. For example, a table of hyperparameters and results for each model you tried would be a good way to summarize your experimentation.

## Marking Scheme
Each of the following will be marked on a 4-point scale:

- Abstract (yes, a succinct summary is important, often this is the only part that gets read!)
- Discussion of prior work, including citations and relevant references
- Appropriate use of visuals and tables
- Description of dataset and preprocessing steps
- Description of model architecture and training process
- Discussion of lessons learned and future work
- Overall clarity and coherence

Resulting in the somewhat weird but arbitrary total of 28 points.

Note that I am not marking you on the performance of your model, but rather your process and description of training and evaluating models.