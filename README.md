# Supervised Machine Learning 
## Credit Risk Classification Analysis
![image](https://www.investopedia.com/thmb/C_bFuBz5TbJphsxLFrlLx3S4zyM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/creditrisk-Final-18f65d6c12404b9cbccd5bb713b85ce4.jpg)

### " Please note the Jupyter notebook "credit_risk_classification.ipynb" is in the folder Credit_Risk as per the Challenge instructions"

## Supervised Machine Learning

The main difference between supervised and unsupervised machine learning is the presence of labeled data. Supervised learning involves training a machine learning model using labeled data, where each data point has a known output or label that the model tries to predict. The goal of supervised learning is to create a model that can accurately predict the output for new, unseen data.

On the other hand, unsupervised learning involves training a model on unlabeled data, where the model tries to find patterns and relationships within the data without being given a specific output to predict. The goal of unsupervised learning is to discover hidden structures or patterns within the data that can be used for various purposes, such as clustering, anomaly detection, or dimensionality reduction.

Supervised learning is a type of machine learning that involves training a model using labeled data. There are two main use cases for supervised learning: classification and regression.

Classification involves predicting a categorical label or class for a given input. For example, a classification model could be trained to classify emails as spam or not spam based on the text content of the email. Other examples of classification tasks include image classification, sentiment analysis, and fraud detection.

Regression, on the other hand, involves predicting a continuous numerical value for a given input. For example, a regression model could be trained to predict the price of a house based on its size, location, and other relevant features. Other examples of regression tasks include stock price prediction, weather forecasting, and demand forecasting.

In both classification and regression, the goal of supervised learning is to create a model that can accurately predict the output for new, unseen data. The quality of the model's predictions is measured using various metrics such as accuracy, precision, recall, and mean squared error.

Resampling strategies are techniques used to address imbalanced datasets in machine learning. When dealing with imbalanced datasets, where one class is significantly more prevalent than another, traditional machine learning algorithms may struggle to make accurate predictions. Resampling strategies can be used to adjust the balance of the dataset and improve model performance.

#### The two main resampling strategies are:

Oversampling: Oversampling involves adding more instances of the minority class to the dataset. This can be done using techniques such as random oversampling, where instances of the minority class are duplicated, or synthetic oversampling, where new instances of the minority class are generated using algorithms such as SMOTE (Synthetic Minority Over-sampling Technique).

Undersampling: Undersampling involves removing instances of the majority class to balance the dataset. This can be done using techniques such as random undersampling, where instances of the majority class are removed at random, or informed undersampling, where instances are removed based on their similarity to instances of the minority class.

Both oversampling and undersampling have their pros and cons. Oversampling can lead to overfitting and may not be effective if the minority class is inherently different from the majority class. Undersampling may lead to loss of important information and may not be effective if the dataset is too small.

In addition to oversampling and undersampling, other resampling strategies include hybrid methods that combine oversampling and undersampling techniques, and cost-sensitive learning, which adjusts the cost of misclassification to reflect the imbalance in the dataset.

Imbalanced datasets are datasets in which the classes are not represented equally. In other words, the number of observations in one class is significantly higher or lower than the number of observations in the other classes. This can be a problem for machine learning algorithms because they are often designed to assume that classes are balanced, and therefore may not perform well on imbalanced datasets.

In the context of binary classification problems, imbalanced datasets typically refer to situations where the number of positive cases (e.g. people who have a disease) is much smaller than the number of negative cases (e.g. people who do not have the disease). This can be a problem because the algorithm may learn to predict the majority class (negative cases) without properly recognizing the minority class (positive cases).

Imbalanced datasets are common in many real-world applications such as fraud detection, spam filtering, and disease diagnosis. Addressing imbalanced datasets often involves resampling strategies, such as oversampling the minority class, undersampling the majority class, or using a combination of both. Other strategies may involve modifying the algorithm itself to adjust for the class imbalance, such as adjusting the decision threshold or using weighted loss functions.


### Overview of the Analysis


The objective of this study is to construct a model that can categorize borrowers' creditworthiness into two distinct groups, using various techniques to train and evaluate the model. The dataset utilized is derived from the lending history of a peer-to-peer lending service company.

Healthy Loans
High-risk Loans

Datasets:

Our original dataset, and
A resampled dataset"

####
