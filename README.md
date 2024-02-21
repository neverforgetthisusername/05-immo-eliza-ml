# Regression

- Repository: `immo-eliza-ml`
- Type: `Consolidation`
- Duration: `3-4 days`
- Deadline: `23/02/2024 5:00 PM`
- Team: solo

## Learning Objectives

- Be able to preprocess data for machine learning.
- Be able to apply a linear regression in a real-life context.

## The Mission

The real estate company Immo Eliza asked you to create a machine learning model to predict prices of real estate properties in Belgium.

After the **scraping**, **cleaning** and **analyzing**, you are ready to preprocess the data and finally build a performant machine learning model!

## Steps

### Data preprocessing

A clean dataset has been prepared in `data/properties.csv`. These are some notes:
- There are about 76 000 properties, roughly equally spread across houses and apartments
- Each property has a unique identifier `id`
- The target variable is `price`
- Variables prefixed with `fl_` are dummy variables (1/0)
- Variables suffixed with `_sqm` indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as `MISSING`

You still need to further prepare the dataset for machine learning. Think about:
- Handling NaNs (hint: **imputation**)
- Converting categorical data into numeric features (hint: **one-hot encoding**)
- Rescaling numeric features (hint: **standardization**)

**Keep track of your preprocessing steps. You will need to apply the same steps to the new dataset later on when generating predictions.** This is crucial! Think _reusable pipeline_!

Additionally, you can also consider (but feel free to skip this and go straight to a first model iteration):
- Preselecting only the features that have at least some correlation with your target variable (hint: **univariate regression** or **correlation matrix**)
- Removing features that have too strong correlation with one another (hint: **correlation matrix**)

Once done, split your dataset for training and testing.

### Model training

The dataset is ready. Let's select a model.

Start with **linear regression**, this will give you a good baseline performance.

If you feel like it, try the more advanced models available via `scikit-learn` as well: Lasso, Ridge, Elastic Net, Random Forest, XGBoost, CatBoost, ... Just refrain from any deep learning model for now.

Train your model(s).

### Model evaluation & iteration

Evaluate your model. What do the performance metrics say?

Don't stop with your first model. Iterate! Try to improve your model by testing different models, or by changing the preprocessing.

## Score board

You should create a model on the training data provided, and test it on your proper test data obtained after splitting the dataset. Your `train.py` script is all yours so configure it how you see fit. You can experiment first in a notebook before you start scripting.

Additionally, we stored around 4000 properties in an external dataset that is secret to you, which we will use as the true out-of-sample test of your model's strength... 😇

The structure of the `predict.py` script needs to stay the same! We configured a CLI tool to point to the external dataset and any output location. The `predict()` function should encapsulate all the logic to go from an input dataset (_with the same structure as the one used for training_) to predictions. Your predictions will be stored and then checked against the actual prices. You can experiment with the script to make sure it works on your end by running following command:
```bash
python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
```

You have three chances (at every day's end) to update your models and the according `predict.py` script. Simply do so by committing the latest changes to your repository.

## Quality Assurance

Read our "Coding Best Practices Manifesto" and apply what's in there!

## Deliverables

1. Publish your code and model(s) on a GitHub repository named `immo-eliza-ml`
    - Follow the structure in the project folder - just copy paste all folders and files to get started
    - Don't forget to do the virtual environment, .gitignore, ... dance

2. Fill in the `MODELSCARD.md` details.

    A **model card** is a document that provides a summary of a model. It is a standardized report that allows to communicate the model's purpose, performance, and limitations. You'd integrate this information into the model validation process where people who did not create the model can check if what you did actually made sense, before your model gets integrated into the company's IT processes.

    Find some (advanced) model card inspiration [here](https://huggingface.co/docs/hub/model-cards).

## Evaluation criteria

| Criteria       | Indicator                                                    | Yes/No |
| -------------- | ------------------------------------------------------------ | ------ |
| 1. Is good     | Your repository is complete                                  | [ ]    |
|                | Your code is clean                                           | [ ]    |
|                | Your models card is clear                                    | [ ]    |
|                | Your `predict.py` script works with the external test data   | [ ]    |
## Quotes

_"Artificial intelligence, deep learning, machine learning — whatever you're doing, if you don't understand it — learn it. Because otherwise you're going to be a dinosaur within 3 years." - Mark Cuban_

![You've got this!](https://media.giphy.com/media/5wWf7GMbT1ZUGTDdTqM/giphy.gif)
