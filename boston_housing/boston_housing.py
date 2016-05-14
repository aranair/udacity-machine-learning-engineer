"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
import pydot
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors

################################
### ADD EXTRA LIBRARIES HERE ###
################################

from scipy import stats
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV


def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    # feature_indexes = range(0,13)
    # for idx in feature_indexes:
    #     feature_name = city_data.feature_names[idx]
    #     pl.figure()
    #     pl.title(feature_name + ' vs Price correlation')
    #     pl.plot(housing_features[:, idx], housing_prices, 'bo')
    #     pl.legend()
    #     pl.xlabel(feature_name)
    #     pl.ylabel('Price')
    #     pl.savefig(feature_name+"_vs_price")

    # prices_graph(housing_prices)
    # boxplot(housing_prices)

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?
    data_size =  housing_prices.size

    # Number of features:
    feature_count = housing_features.shape[1]

    # Minimum price?
    min_price = np.nanmin(housing_prices)

    # Maximum price?
    max_price = np.nanmax(housing_prices)

    # Calculate mean price?
    mean_price = np.mean(housing_prices)

    # Calculate median price?
    median_price = np.median(housing_prices)

    # Calculate standard deviation?
    standard_deviation = np.nanstd(housing_prices)

    print "Data size: " + str(data_size)
    print "Feature count: " + str(feature_count)
    print "Minimum price: " + str(min_price)
    print "Maximum price: " + str(max_price)
    print "Mean Price: " + str(mean_price)
    print "Median Price: " + str(median_price)
    print "Standard Deviation: " + str(standard_deviation)


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################

    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return mean_squared_error(label, prediction)

def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # if depth >= 4 and depth <= 6:
    #     pl.figure()
    #     pl.plot(y_test, 'bo')
    #     pl.plot(regressor.predict(X_test), color='red')
    #     pl.savefig("test_data_depth_" + str(depth))

    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err, depth)

def boxplot(prices):
    """Plot boxplot of housing prices."""
    pl.boxplot(prices)
    pl.savefig('boxplot')

def prices_graph(prices):
    """Plot housing prices."""
    prices = sorted(prices)
    fit = stats.norm.pdf(prices, np.mean(prices), np.std(prices))
    pl.plot(prices ,fit, '-o')
    pl.hist(prices, normed=True)
    pl.savefig('prices_distribution')

def learning_curve_graph(sizes, train_err, test_err, depth):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.savefig("learning_curve_depth_" + str(depth))
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.savefig("model_complexity")
    pl.show()

def find_nearest_neighbor_indexes(x, X):
   neigh = NearestNeighbors(n_neighbors = 10)
   neigh.fit(X)
   distance, indexes = neigh.kneighbors(x)
   return indexes

def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {
        'max_depth': (1,2,3,4,5,6,7,8,9,10)
    }

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    mean_squared_error_scorer = make_scorer(performance_metric, greater_is_better = False)

    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
    grid = GridSearchCV(
            estimator = regressor,
            param_grid = parameters,
            scoring = mean_squared_error_scorer,
            n_jobs = -1)

    grid.fit(X, y)

    print "Best Params: " + str(grid.best_params_)

    regressor.set_params(**grid.best_params_)

    # Fit the learner to the training data to obtain the best parameter set
    print "Final Model: "
    regressor.fit(X, y)

    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = regressor.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)

    indexes = find_nearest_neighbor_indexes(x, X)
    sum_prices = []
    for i in indexes:
        sum_prices.append(city_data.target[i])

    neighbor_avg = np.mean(sum_prices)
    print "Nearest Neighbors average: " + str(neighbor_avg)

# In the case of the documentation page for GridSearchCV, it might be the case that the example is just a demonstration of syntax for use of the function, rather than a statement about
def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
