import numpy as np
import util
import sys
import matplotlib.pyplot as plt


sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression
from decision_analysis import plot_decision_line

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    x_train, y_train = util.load_dataset(train_path, add_intercept=True,
            label_col='t')
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True,
            label_col='t')
    from logreg import LogisticRegression
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print(clf.theta)

    fig, ax = plt.subplots(1,1,figsize=(12,8))

    ax.scatter(x_valid[:,1], x_valid[:,2],c=y_valid.astype(np.int))
    ax.set_ylim(x_valid[:,2].min(), x_valid[:,2].max())
    plot_decision_line(clf.theta, x_valid, ax)
    plt.savefig("posonly_all_observed.png")
    plt.show()


    



    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # Part (b): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, add_intercept=True,
            label_col='y')
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True,
            label_col='y')
    from logreg import LogisticRegression
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print(clf.theta)

    fig, ax = plt.subplots(1,1,figsize=(12,8))

    ax.scatter(x_valid[:,1], x_valid[:,2],c=y_valid.astype(np.int))
    ax.set_ylim(x_valid[:,2].min(), x_valid[:,2].max())
    plot_decision_line(clf.theta, x_valid, ax)
    plt.savefig("naive_training_partial.png")
    plt.show()
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    #decition 
    y_pred =clf.predict(x_valid)
    print(y_pred)

    fig, ax = plt.subplots(1,1,figsize=(12,8))

    ax.scatter(x_valid[:,1], x_valid[:,2],c=y_valid.astype(np.int))
    ax.set_ylim(x_valid[:,2].min(), x_valid[:,2].max())
    plt.show()

    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
