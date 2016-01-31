## Project 1: Predicting Boston Housing Prices

### 1) Statistical Analysis and Data Exploration

There are a total of 506 houses in the dataset, with 13 different features. The minimum housing price data point is 5.0, which is equivalent to $5000 and the maximum price data point of $50000. The mean housing price is approximately $22532.81 and the median price $21200, with a standard deviation of approximately $9188.01.


### 2) Evaluating Model Performance

- Which measure of model performance is best to use for predicting Boston housing data and analyzing the errors? Why do you think this measurement most appropriate? Why might the other measurements not be appropriate here?
- Why is it important to split the Boston housing data into training and testing data? What happens if you do not do this?
- What does grid search do and why might you want to use it?
- Why is cross validation useful and why might we use it with grid search?

I have chosen to use the median absolute error to measure the model performance.

I checked the data with the box plot technique, and found quite a high number of outliers in the data points (IQR: 7.975, Q1: 17.025, Q3: 25.0). 

Median absolute error measurement method is more robust and is more resistant against the outliers.

Mean squared error is a commonly used measurement for regression models but because of the count of outliers that will skew the function towards these outliers, it is a less appropriate measurement to take. 

To find the best estimator paramaters for the decision tree regression model.

http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data.

This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally.




### 3) Analyzing Model Performance

- Look at all learning curve graphs provided. What is the general trend of training and testing error as training size increases?
- Look at the learning curves for the decision tree regressor with max depth 1 and 10 (first and last learning curve graphs). When the model is fully trained does it suffer from either high bias/underfitting or high variance/overfitting?
- Look at the model complexity graph. How do the training and test error relate to increasing model complexity? Based on this relationship, which model (max depth) best generalizes the dataset and why?

### 4) Model Prediction

- Model makes predicted housing price with detailed model parameters (max depth) reported using grid search. Note due to the small randomization of the code it is recommended to run the program several times to identify the most common/reasonable price/model complexity.
- Compare prediction to earlier statistics and make a case if you think it is a valid model.
