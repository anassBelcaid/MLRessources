"""
Personal script for a better understanding of the classification problem
"""


import numpy as np
import matplotlib.pyplot as plt
from util import load_dataset,plot
from pathlib import Path
from poisson import PoissonRegression



if __name__ == "__main__":
    
    #path for the datasets
    train_path = Path("train.csv")
    valid_path = Path("valid.csv")
    test_path = Path("test.csv")

    #loading the datasets
    x_train, y_train = load_dataset(train_path, add_intercept=1)
    x_valid, y_valid = load_dataset(valid_path, add_intercept=1)
    x_test, y_test = load_dataset(test_path, add_intercept=1)

    num_features = x_train.shape[1]

    #Regression
    clf = PoissonRegression(step_size=1e-6,theta_0= np.random.rand(num_features))
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    plot(x_train, y_train, clf.theta, "poisson_glm.png")
    plt.show()

    print("Mean square error = {:e}".format(np.mean(y_hat**2 - y_test**2)))





