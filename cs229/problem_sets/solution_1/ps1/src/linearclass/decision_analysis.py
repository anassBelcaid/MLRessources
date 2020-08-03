"""
Script to plot the decision boundaries and the difference 
between Linear classifier and Gaussian Descriminant analysis
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap']='Paired'
import util
from pathlib import Path
from logreg import LogisticRegression
from gda import  GDA


def plot_decision_line(theta, x_valid, ax, color='r', label='decision line'):
    """
    plot the decision line on x_valid
    """

    m, M = x_valid[:,1].min(), x_valid[:,1].max()

    x0 = np.linspace(m, M)

    x1 = (0.5 - theta[0] - theta[1]*x0)/theta[2]

    ax.plot(x0,x1, color, label=label)

if __name__ == "__main__":
    
    train_path = Path("./ds2_train.csv")
    valid_path = Path("./ds2_valid.csv")
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)


    #training the logistic regression
    lreg = LogisticRegression()
    lreg.fit(x_train, y_train)


    #GDA
    gda = GDA()
    gda.fit(x_train[:,[1,2]], y_train)





    #plotting the dataset
    fig, ax = plt.subplots(1,1, figsize=(12,8)) 
    plot_decision_line(lreg.theta, x_valid[:,[1,2]], ax, label='lin reg',
            color='C0')

    plot_decision_line(gda.theta, x_valid[:,[1,2]], ax, label='gda',
            color='C1')
    plt.scatter(x_valid[:,1], x_valid[:,2], c=y_valid.astype(np.int),
            label='')
    plt.title("Decision margin on dataset 1")
    plt.legend()
    plt.savefig("com_gda_logreg_ds2.png")
    plt.show()
