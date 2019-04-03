#import files

#import csv
import csv

#import numpy

import numpy as np

#import SVR
from sklearn.svm import SVR

#import plt

import matplotlib.pyplot as plt

dates = []


# prices array

prices = []

def get_data(filename):

    with open(filename, 'r') as csvfile:

        csvfilereader = csv.reader(csvfile)

        print('')

        next(csvfilereader)


        for row in csvfilereader:

            dates.append(int(row[0].split('/')[2]))

            prices.append(float(row[1]))

    return


def predict_prices(dates, prices):

    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel = 'linear', C = 1e3)

    svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)

    svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma=0.1)


    svr_lin.fit(dates, prices)

    svr_poly.fit(dates, prices)

    svr_rbf.fit(dates, prices)



    plt.scatter(dates, prices, color='black', label='Data')

    # 1) RBF model

    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')

    # 2) Linear model

    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')

    # 3) Poly model

    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Poly model')


    plt.xlabel('Date')

    print('')

    plt.ylabel('Prices')

    plt.title('Support Vector Regression')

    plt.legend()

    print('')

    plt.show()

    return


print ('')

get_data(r'C:\Users\avina\PycharmProjects\Stock\aapl.csv')

print ('')

predicted_price = predict_prices(dates, prices)

print ('')