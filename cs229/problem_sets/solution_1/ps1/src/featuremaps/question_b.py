"""
Code a three degree polynomial
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(path, target_label='y'):
    """
    simple load the csv
    """

    train = pd.read_csv(path)

    target = train[target_label]

    train.drop(target_label, axis=1, inplace=True)


    return train, target


if __name__ == "__main__":


    x_train , y_train = load_data("./train.csv", target_label='y')
    x_valid , y_valid = load_data("./valid.csv", target_label='y')

    for p in range(2,4):
        x_train['x'+str(p)] = x_train['x']**p
        x_valid['x'+str(p)] = x_valid['x']**p


    #model 
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_valid)
    x_train = x_train.values
    x_valid = x_valid.values

    plt.plot(x_valid[:,0], y_valid,'bo',ms=2)
    plt.plot(x_valid[:,0], y_valid,'r+',ms=4)
    plt.show()


