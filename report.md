## Project 1: Predicting Boston Housing Prices

### 1) Statistical Analysis and Data Exploration

There are a total of 506 houses in the dataset, with 13 different features. The minimum housing price data point is 5.0, which is equivalent to $5000 and the maximum price data point of $50000. The mean housing price is approximately $22532.81 and the median price $21200, with a standard deviation of approximately $9188.01.


### 2) Evaluating Model Performance

- **Which measure of model performance is best to use for predicting Boston housing data and analyzing the errors? Why do you think this measurement most appropriate? Why might the other measurements not be appropriate here?**
I have chosen to use the median absolute error to measure the model performance. I have checked the data with a box plot and found quite a high number of outliers in the data points (IQR: 7.975, Q1: 17.025, Q3: 25.0). And out of all the measurement methods, this one is known to be more robust and more resistant against the outliers, therefore it is a good choice of model performance measurer in this use case.

Mean squared error is a commonly used measurement for regression models but because of the count of outliers, it will skew the model towards these outliers. It is a less appropriate measurement to take. 

- **Why is it important to split the Boston housing data into training and testing data? What happens if you do not do this?**
The thinking underlying such a split is that we can treat learning data as "present" data and test data "future" data. Holding back some of the present data for testing is therefore a fair approximation to having future data for testing. In other words, we can use these test data to provide honest assessments of the performance of the predictive model.  

Without the split, and with the entire set of available data being fed as training data into the model; the model would learn thwould then learn the training data perfectly yet would probably fail to predict anything useful on unseen data, or otherwise known as overfitting.

- **What does grid search do and why might you want to use it?**
Grid search is an exhaustive search over different possible combinations of parameter values for a model using K-fold cross-validation techniques. It cross-validate the results as it goes through every parameter combination to determine which set of parameter values gives us the best performance for a particular model. It is a systematic way to select a set of hyperparameters for the model and in this case, it is the max depth of the decision tree regressor.


- **Why is cross validation useful and why might we use it with grid search?**
Cross-validation helps us to estimate how well the model will perform when it is asked to make predictions for unseen data, in other words, how well it generalises.  The grid search employs this to evaluate each parameter combination. In specific, the K-fold cross-validation technique is used whereby the dataset is divided into K subsets, and each of the K subsets would be taken as test data to evaluate one parameter combination (running K times), and the average error would be used for the evaluation of that one set of parameter combination.


### 3) Analyzing Model Performance

- **Look at all learning curve graphs provided. What is the general trend of training and testing error as training size increases?**
The error rate tends to converge quickly at first with the first 50 data points. However, beyond 50 data sets, the training error usually continues to decrease with the test error remaining high.

- **Look at the learning curves for the decision tree regressor with max depth 1 and 10 (first and last learning curve graphs). When the model is fully trained does it suffer from either high bias/underfitting or high variance/overfitting?**
Past the optimal depth found by the GridSearch, it suffers from overfitting when the model is fully trained. The training error reduces to 0 which means that the model is adapting to the training data more and more yet stagnating on the performance of predicting unseen test data. This is clearly a sign of **overfitting**.

- **Look at the model complexity graph. How do the training and test error relate to increasing model complexity? 
Based on this relationship, which model (max depth) best generalizes the dataset and why?**
They are both reduced up to about max-depth of 5. Beyond that, only the training error continues to decrease while the test error remains stagnant. Based solely on this relationship, just by looking at this one particular graph, one may argue that since the total error does not go up as the model complexity increases, there may be a false conclusion that the higher the max-depth, the better the model. 

- **However**, we know that there is always a bias-variance trade off; otimizing the model by minimizing variance will inevitably increase bias and vice-versa. Minimizing bias while ignoring variance will yield a model that would be highly inaccurate (even if it averages out to be correct in some sense) and the converse would yield a model that is systematically wrong. Understanding this, we can then draw the conclusion that the optimal depth would be when the test data stagnates, because beyond that further minimizing the bias (via training data) is no longer helping the model. The balance would be at this point.

### 4) Model Prediction

- **Model makes predicted housing price with detailed model parameters (max depth) reported using grid search. Note due to the small randomization of the code it is recommended to run the program several times to identify the most common/reasonable price/model complexity.**

With max depth of 5, the predicted price would be approximately USD20977.63. (20.96776316)

- **Compare prediction to earlier statistics and make a case if you think it is a valid model.**
