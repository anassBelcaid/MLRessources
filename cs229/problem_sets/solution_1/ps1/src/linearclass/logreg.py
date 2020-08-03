import numpy as np
import util
import matplotlib.pyplot as plt
plt.xkcd()


def sigmoid(x):
    return 1./(1+np.exp(-x))

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    cls = LogisticRegression()
    cls.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    prediction = cls.predict(x_valid)
    plt.scatter(x_valid[:,1], x_valid[:,2], c=y_valid, cmap='Paired')

    #plotting the decision line
    m, M = x_valid[:,1].min(), x_valid[:,1].max()
    x_dec = np.linspace(m, M, 100)
    y_dec = (0.5 - cls.theta[0] - cls.theta[1]*x_dec)/cls.theta[2]
    plt.plot(x_dec, y_dec, 'C0')
    plt.title("Logistic classification")
    plt.savefig("classificaiton_1.png")
    plt.show()

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, prediction)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n = x.shape[0]
        self.theta = np.zeros(x.shape[1])

        new_theta = self.NewtonStep(x, y)
        error =np.linalg.norm(self.theta - new_theta) 

        while  error > self.eps:

            #computing the new theta
            self.theta = new_theta

            new_theta = self.NewtonStep(x, y)

            #computing the new error
            error = np.linalg.norm(self.theta - new_theta)
            # print("Loss = {:.6f}".format(self.Loss(x,y)))




    def NewtonStep(self, x, y):
        """
        Perform a newton step to compute the minimal value for the loss function
        """
        n = x.shape[0]
        scores = x.dot(self.theta)

        D = np.diag(sigmoid(scores)*(1-sigmoid(scores)))
        Hessian = 1/n * x.T.dot(D).dot(x)

        #Gradient
        grad = -1/n*x.T.dot(y - sigmoid(scores)) 


        loss = self.Loss(x, y)


        return self.theta - np.linalg.inv(Hessian).dot(grad)





    def Loss(self,x, y):
        """
        Compute the loss with actual data for better
        debugging possibilities
        """

        scores = sigmoid(x.dot(self.theta))

        return - np.mean( y*np.log(scores) + (1-y)*np.log(1-scores))



    
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return sigmoid(x.dot(self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
