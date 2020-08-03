import numpy as np
import util
import matplotlib.pyplot as plt
plt.xkcd()

def sigmoid(x):

    return 1/(1+ np.exp(-x))

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    cfl = GDA()
    cfl.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

    y_hat = cfl.predict(x_valid)
    plt.scatter(x_valid[:,0], x_valid[:,1], c=y_valid.astype('int'),
            cmap='Paired')
    # plot the decision boundary
    x_decision = np.linspace(x_valid[:,0].min(), x_valid[:,0].max())
    y_decision = (0.5 - cfl.theta[0]*x_decision - cfl.theta[2])/cfl.theta[1]
    plt.plot(x_decision, y_decision, 'b')
    plt.xlim(x_valid[:,0].min(), x_valid[:,0].max())
    plt.ylim(x_valid[:,1].min(), x_valid[:,1].max())
    plt.title("Decision GDA")
    plt.savefig("decision.gda.png")
    plt.show()
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        mask_1 = (y==1)
        mask_0 = (y==0)
        phi = mask_1.mean()

        mu_1 = x[mask_1,:].mean(axis=0)
        mu_0 = x[mask_0,:].mean(axis=0)

        #compute centred data
        x_cent = x.copy()
        x_cent[mask_0,:] -= mu_0
        x_cent[mask_1,:] -= mu_1

        sigma = 1/x.shape[0] * x_cent.T.dot(x_cent)

        self.theta = np.zeros(3)
        self.theta[:2] = (mu_0 - mu_1).T.dot(np.linalg.inv(sigma))
        self.theta[2]  = (mu_0 - mu_1).T.dot(np.linalg.inv(sigma)).dot(mu_0 -
                mu_1)


        #computing alpha
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = x.dot(self.theta[:2]) + self.theta[2]
        return (z>0.5)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
