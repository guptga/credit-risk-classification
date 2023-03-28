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

Borrowers: 

Healthy Loans

High-risk Loans

Datasets:

Our original dataset

A resampled dataset

#### Steps

To analyze loan data, we follow a process that includes splitting the data into training and testing sets, where the "loan_status" column serves as our target outcome ("y") and the other columns as our features ("x"). A "loan_status" value of 0 indicates a healthy loan, while a value of 1 indicates a high risk of defaulting.

Initially, we create a Logistic Regression model using the original dataset, which contains 75036 records with a "loan_status" of healthy and 2500 records with a high risk of defaulting. We evaluate the model's performance using accuracy score, confusion matrix, and classification report metrics.

Next, we use the RandomOverSampler module from the imbalanced-learn library to resample the original data and generate two equal sets of 56277 records each, representing healthy and high-risk loans. We then rerun the Logistic Regression model on the resampled data and evaluate its performance using the same metrics as before.


### Results

#### Machine Learning Model 1:

* When assessing the performance of a binary classifier, balanced accuracy is a useful metric to consider, particularly when the classes are imbalanced. Imbalance occurs when one class occurs more frequently than the other, which is common in scenarios such as anomaly detection. Balanced accuracy is calculated as the average recall obtained on each class.

* Although a high score for the model's balanced accuracy and predictions suggests a high degree of confidence in the model's performance, it does not provide a complete picture. Other factors, such as precision, false positive rate, and true negative rate, should also be taken into account when evaluating the classifier's effectiveness.

###### Training Data Score: 0.9914878250103177
###### Testing Data Score: 0.9924164259182832
###### Balanced Accuracy Score: 0.9442676901753825

* Upon reviewing the confusion matrix, we observe that the model accurately predicted 18678 loans as "healthy" and 558 loans as "high-risk." While this indicates a reasonable degree of accuracy and precision when compared to the false positive and false negative rates, there is room for improvement in minimizing these errors.


* The classification report provides additional insight into the model's performance. Precision and recall values are particularly relevant because misclassifying data can have significant consequences. A high precision score is crucial for reducing false positives, which could result in a loss of potential customers. Conversely, a high recall score is essential for minimizing false negatives, which could lead to significant financial losses.

* The logistic regression model we used achieved high accuracy (0.99). When predicting healthy loans, recall produced the best overall predictive outcome, with the highest true positive rate for healthy loans and true negative rate for high-risk loans. However, this result is potentially misleading because our dataset is imbalanced. Creating a more balanced dataset through resampling may lead to a higher level of precision.
![image](https://user-images.githubusercontent.com/116124534/228239804-fb46e807-8590-4c35-ada1-3304b5145a25.png)

#### Machine Learning Model 2:

* Balanced accuracy is a metric that indicates how well a binary classifier performs when the number of samples in each class is roughly the same. If the dataset is imbalanced, a model can achieve high accuracy by simply predicting the majority class most of the time, while ignoring the minority class, which is often the class of interest. For instance, if the majority class accounts for 99% of the data, a simple algorithm such as logistic regression will struggle to identify the minority class data points.

In our analysis, high-risk loans may be the minority class. Resampling is a technique that involves either under-sampling the majority class or over-sampling the minority class. In over-sampling, new samples are generated by duplicating random records from the minority class. RandomOverSampler is an implementation of this technique that samples with replacement.

By resampling our data, we can mitigate the effects of class imbalance and improve the accuracy of our predictions. In fact, resampling can potentially increase the accuracy to nearly 100%.


###### Training Data Score: 0.9941016646031091
###### Testing Data Score: 0.9952022286421791
###### Balanced Accuracy Score: 0.9959744975744975

* The confusion matrix of our resampled dataset reveals that the true positives (healthy loans = 18678) have slightly decreased compared to our original dataset, while the true negatives (high-risk loans = 558) have increased. Although resampling has enhanced our capacity to predict high-risk loans, it has also resulted in more healthy loans being excluded and mislabeled as high-risk.

* The classification report provides additional insights on the performance of the logistic regression model. The model achieved outstanding accuracy (1.00), which suggests that it performed very well in distinguishing between healthy loans and high-risk loans. 

* When predicting healthy loans, both precision and recall were excellent, resulting in no false positives or negatives. Moreover, the model achieved the highest true positives (healthy loans) and true negatives (high-risk loans) for healthy loans. However, for high-risk loans, precision remained unchanged after resampling the data. This finding indicates that training the model on synthetic data may affect its performance when tested against new data.

![image](https://user-images.githubusercontent.com/116124534/228241679-a32c00ac-4fb1-441c-ad5c-570f24e0dde9.png)


### Summary

Resampling the data improves model performance on test data with strong precision and perfect recall. The resampled data improves the initial Logistic Regression model with an accuracy score of 100% and a higher balanced accuracy score. A lending company might favor a model that favors the ability to predict high-risk loans. However, oversampling may generate synthesized samples that do not belong to the minority class, resulting in incorrect predictions in the real world. One option is to undersample the majority class to decrease the number of overrepresented values in the dataset.
