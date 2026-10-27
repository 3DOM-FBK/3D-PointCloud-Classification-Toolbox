# Random Forest Training configuration and Metrics

## Training Configuration

<a href="../classifyViewer/viewer/static/viewer/images/Training_configuration.png" target="_blank"> <img src="../classifyViewer/viewer/static/viewer/images/Training_configuration.png" alt="Training configuration" width="700"> </a>

In the training configuration, you can decide the model name and the data split of the two sets as showned in the picture.
In addition, it is possible to choose which features to select during the training (<code>RGB</code>  can be included using the button below)
Morever, you can decide the parameters values for RF model, which are:
* n of estimators: number of trees in the forest (<code>default: 200</code>); 
* max depths: maximum depth of each tree (<code>default: 15</code>);
* min samples split: minimum number of samples required to be at a split node (<code>default: 20</code>); 
* n of jobs: number of threads used to train the model (used only with CPU, <code>default: 12</code>);
* use gpu: whether to use <code>GPU</code>  for training you can flag the button (<code>default: "off"</code>); 


## Training Results

<a href="../classifyViewer/viewer/static/viewer/images/Training_results.png" target="_blank"> <img src="../classifyViewer/viewer/static/viewer/images/Training_results.png" alt="Training results" width="700"> </a>

This viewer window explain the report with different sections: model evaluation metrics, sorted feature importance, and different processing time for each steps.

### Model Evaluation Metrics

The following metrics are used to evaluate the performance of the model:

- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html" target="_blank" rel="noopener noreferrer">Overall Accuracy</a>: measures the proportion of correctly classified samples over the total number of samples.

- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html" target="_blank" rel="noopener noreferrer">Precision</a>:

  $\displaystyle\text{Precision} = \frac{TP}{TP + FP}$

- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html" target="_blank" rel="noopener noreferrer">Recall</a>:

  $\displaystyle\text{Recall} = \frac{TP}{TP + FN}$

- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html" target="_blank" rel="noopener noreferrer">F1-score</a>:

  $\displaystyle\text{F1-Score} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$

- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html" target="_blank" rel="noopener noreferrer">Intersection over Union (IoU)</a>

The accuracy metrics computed for each class represent the performance of the model in classifying samples belonging to that specific class.

### Feature Importance

The sorted feature importance values represent the features ranked according to their importance in the model. The sum of the importance values of all features should equal `1.0`. For instance, if the model has 5 features, each feature will have an importance value between 0 and 1, representing its relative contribution to the model.The sum of the importance values of all 5 features will be `1.0`. For simplicity, the report visualizes only the most important features, up to a cumulative importance of `0.91`. The remaining features are hidden, while the overall sum of all feature importance values remains `1.0`.

### Processing Time

The processing time is reported for each of the following steps:

- **Loading data:** The time required to load the data into memory.
- **Training the model:** The time required to train the model.
- **Computing metrics:** The time required to compute the evaluation metrics.
- **Overall time:** The total processing time across all steps.
