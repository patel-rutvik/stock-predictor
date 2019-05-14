# using the site: https://towardsdatascience.com/simple-stock-price-prediction-with-ml-in-python-learners-guide-to-ml-76896910e2ba
import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from TFANN import ANNR
#from google.colab import files

#files.upload()

def getData():
    #reads data from the file and ceates a matrix with only the dates and the prices 
    stock_data = np.loadtxt('AAPL.csv', delimiter=",", skiprows=1, usecols=(1, 4))
    #scales the data to smaller values
    stock_data=scale(stock_data)
    #gets the price and dates from the matrix
    prices = stock_data[:, 1].reshape(-1, 1)
    dates = stock_data[:, 0].reshape(-1, 1)
    #creates a plot of the data and then displays it
    mpl.plot(dates[:, 0], prices[:, 0])
    mpl.show()
    return

if __name__ == "__main__":
    while(True):
        userIn = int(input("Please choose one of the integer options:\n 1) setup\n 2) train\n 3) test\n 4) Quit\n"))
        if (userIn == 1):
            getData()
        elif (userIn == 2):
            pass
        elif (userIn == 3):
            pass
        elif (userIn == 4):
            print("Terminating...")
            break
        else:
            print("Please enter a valid input...")
            pass
    pass

'''
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def fetch_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.appent(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
        return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, len(dates, 1))

    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)

    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color = 'black', label = 'Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.xlabel('DATE')
    plt.ylabel('PRICE')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return svr_rbf.predict(x)[0]

fetch_data('aapl.csv')
predicted_price = predict_prices(dates, prices, 29)

print(predicted_price)
'''
