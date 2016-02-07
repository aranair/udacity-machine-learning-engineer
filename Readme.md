# 1) Statistical Analysis and Data Exploration

- Data size: 506 houses
- Feature count: 13
- Minimum price: 5.0 ($5000)
- Maximum price: 50.0 ($50000)
- Mean Price: 22.5328063241 (~$22532.81)
- Median Price: 21.2 (~$21200)
- Standard Deviation: 9.18801154528 ($9188.01)

# 2) Evaluating Model Performance

## Measures of model performance
**Which measure of model performance is best to use for predicting Boston housing data and analyzing the errors? Why do you think this measurement most appropriate? Why might the other measurements not be appropriate here?**

The mean squared error would be the most appropriate measurement for this set of data for the analysis of the errors. The main reason is that MSE is more sensitive, penalising huge differences more at the same time. This helps us to detect even small differences in performance. On top of that, the decision tree regressor also uses MSE as the split criterion while building the model so MSE would be a sound choice here.

If we had used the median absolute error (MAE) for example, it is possible that we might not be able to tell apart the performance between parameters with errors that have similar medians, even if the overall performance using one set is better is better than the other. MAE would be suitable if we wanted to eliminate outliers while actually building a different model.

Technically, we could use the mean absolute error, but MSE would be a better pick because of sensitivity.

## Splitting the data
**Why is it important to split the Boston housing data into training and testing data? What happens if you do not do this?**

How well a model performs is equivalent to how well it can predict future unseen data. We need some way to do this evaluation. The rationale that underlies the data split is that we can treat training data as "present" data and test data as "future" data, assuming that the trends remain the same. Holding back some of the present data for testing would be a fair method to having future data for testing. In other words, we can then use these test data to provide an objective assessment of the performance of the predictive model.  

Without the split, and with the entire set of available data being fed as training data into the model; the model would then learn the training data perfectly yet would be incapable of predicting well on unseen data, an overfitted model.

--
**What does grid search do and why might you want to use it?**

Grid search is an exhaustive search over different possible combinations of hyperparameter values for a model. It cross-validates the results as it loops through every parameter combination to determine which set of parameter values gives us the best performance for a particular model. It is a systematic way to fine-tune or select a set of hyperparameters for the model that enables it to perform the best according to some performance metrics. The systematic and exhaustive nature of the grid search basically guarantees that the best set of parameters will be found, assuming the correct metrics is used (even if it would take a long time for bigger datasets).

--
**Why is cross validation useful and why might we use it with grid search?**

Cross-validation is a way for us to estimate how well the model will perform when it is asked to make predictions for unseen data, in other words, how well it generalises. The grid search uses this to evaluate each parameter combination. In specific, the K-fold cross-validation technique is used whereby the dataset is divided into K partitions (in this case I've chosen 10), and each of the K partitions would be taken as test data to evaluate one parameter combination (running K times), and the average MSE taken for each parameter would be used for evaluation.



# 3) Analyzing Model Performance

**Look at all learning curve graphs provided. What is the general trend of training and testing error as training size increases?**

Training error starts off low while testing error starts off high because the model has not generalised yet and it is basing predictions off of just training data (which explains the low training error). Generally, the error rates would then start to converge as the training size increases. For different depths, the graph of the training errors differs slightly depending on whether it is a badly underfitted model (max depth 1) where training error increases to a significant level quickly with dropping test errors or if it is a overfitted model (max depth of 10) where training error never really rise much because the model is very well molded to the training set.

--

**Look at the learning curves for the decision tree regressor with max depth 1 and 10 (first and last learning curve graphs). When the model is fully trained does it suffer from either high bias/underfitting or high variance/overfitting?**

We see two extremes for max depth of 1 vs 10. 

At max depth of 1, it is clearly underfitted as the training errors are still relatively high and the total errors show no signs of dropping despite more data being fed into the model.

Beyond the optimal depth (~5), the model starts to suffer from overfitting when the model is fully trained. The training error rate slowly closes in to 0 which means that the model is adapting to the training data extremely well yet at the same time it stagnates in its ability to predict unseen test data. At max depth of 10, the training error stays extremely low, almost zero with the training errors remaining high. This is clearly a sign of **overfitting**.

--

**Look at the model complexity graph. How do the training and test error relate to increasing model complexity? 
Based on this relationship, which model (max depth) best generalizes the dataset and why?**

They are both reduced up to about max-depth of 5. Beyond that, only the training error continues to decrease while the test error remains stagnant. Based solely on this relationship, just by looking at this one particular graph, one may argue that since the total error does not go up as the model complexity increases, there may be a false conclusion that the higher the max-depth, the better the model. 

**However**, there is always a bias-variance trade off; otimizing the model by minimizing variance will inevitably increase bias and vice-versa. Minimizing bias while ignoring variance will yield a model that would be highly inaccurate (even if it averages out to be correct in some sense) and the converse would yield a model that is systematically wrong. Understanding this, we can then draw the conclusion that the optimal depth would be when the test data stagnates, because beyond that further minimizing the bias (via training data) is no longer helping the model.

# 4) Model Prediction

## Prediction 
**Model makes predicted housing price with detailed model parameters (max depth) reported using grid search. Note due to the small randomization of the code it is recommended to run the program several times to identify the most common/reasonable price/model complexity.**

The optimal max depth found by grid search for multiple runs was between 4 to 6. With a max depth of 4, the price found would be ~21629.74. With a max depth of 6, the predicted price would be ~20766. So a price prediction between 20766 to 21629.74 would be a fair estimation.

**Compare prediction to earlier statistics and make a case if you think it is a valid model.**

By examining the various feature's vs price graphs, we can generally see that there are a number of strongly correlated features such as CRIM (per capita crime rate by town), RM (average number of rooms per dwelling), LSTAT (% lower status of the population) and DIS (weighted distances to five Boston employment centres) and others. 

Also, judging from the graphs, CRIM being higher should indicate a price drop, RM being lower may mean a potential price drop as well, and similarly for DIS, which true enough is reflected in the predicted price between 20766 to 21629.74. It is well within 1 std of the mean price.

While the prediction of this one house may be fairly accurate on this one house, I do not feel that the model properly assigns weight to certain features that may have stronger correlations with the price, nor decrease the weight for lesser correlated features (nor remove for that matter). The decision tree also does not take into account inter-correlated features, and therefore may not perform well when certain features are evaluated. I conclude that perhaps the model is valid to some extent, yet insufficiently accurate.

```
Features - means and standard deviation:

CRIM: 3.59376071146 (std: 8.58828354765) [Predicting house: 11.95]
ZN: 11.3636363636 (std: 23.2993956948)
INDUS: 11.1367786561 (std: 6.85357058339)
CHAS: 0.0691699604743 (std: 0.25374293496)
NOX: 0.554695059289 (std: 0.115763115407)
RM: 6.28463438735 (std: 0.701922514335) [Predicting house: 5.609]
AGE: 68.5749011858 (std: 28.1210325702)
DIS: 3.79504268775 (std: 2.10362835634) [Predicting house: 1.385]
RAD: 9.54940711462 (std: 8.69865111779)
TAX: 408.23715415 (std: 168.370495039)
PTRATIO: 18.4555335968 (std: 2.16280519148)
B: 356.674031621 (std: 91.2046074522)
LSTAT: 12.6530632411 (std: 7.13400163665) [Predicting house: 12.13]
Mean Housing Price: 22.53 (std: 9.18801154528)
```

