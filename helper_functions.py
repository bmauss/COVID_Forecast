import warnings
warnings.filterwarnings('ignore')
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import math
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set_style('darkgrid')
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf



def acf_pacf(series, alags=120, plags=22):
    '''
    Input: Time Series, and the number of lags for Autocorrelation 
    and Partial Autocorrelation.
    
    Outputs: ACF and PACF plots for time series
    '''
    
    #Create figure
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(13,8))
    #Make ACF plot
    plot_acf(series,lags=alags, zero=False,ax=ax1)
    #Make PACF plot
    plot_pacf(series,lags=plags, ax=ax2)
    plt.show()
    
def decompose_time_series(series):
    """
    Input a series
    
    Decompose a time series and plot it
    
    Outputs: 
        Decomposition plot
    """
    
    result = seasonal_decompose(series)
    fig = result.plot()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.show();

def train_test(df):
    '''
    Input a dataframe.
    
    Function resets the index, and calculates the proportion of data to 
    allocate to the respective datasets.  It then creates the datasets by 
    indexing the values.  Finally, both datasets have the index set back to 
    'date'.
    
    Output: Train and Test sets for Time Series Analysis
    '''
    
    # Reset Index
    temp = df.reset_index()
    
    # Training set test size
    train_size = int(round(len(df.index) * 0.80, 0))
    
    # Set training data
    train = temp.iloc[:train_size]
    
    # Set test data
    test = temp.iloc[train_size:]
    
    # Set index back to 'date'
    train = train.set_index('date')
    test = test.set_index('date')
    
    # Set Frequency to Days
    train = train.asfreq('D')
    test = test.asfreq('D')
    return train, test

def preprocess_data(df, column):
    '''
    Input DataFrame and column name
    
    Function will create a numpy array from the values and set them to float.
    The values will be reshaped and normalized.  Next the dataset will be 
    split into training, validation, and test sets.
    
    Returns: Training, Validation, and Test sets
    '''
    
    # Instantiate scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reset Index
    df = df.reset_index()
    
    # Create a series of only the values of the specified columns
    df = df[column].values
    
    # Reshape and convert to numpy array
    df = np.reshape(df, (-1, 1))
    
    # Normalize data
    df = scaler.fit_transform(df)
    
    # Define stopping points for Train and Validation Sets 
    train_stop = int(len(df) - 21)
    val_stop = int(train_stop + 7)
    
    # Define indices for train, val, and test
    train, val, test = df[0:train_stop,:], df[train_stop:val_stop,:], df[val_stop:,:]
    
    return train, val, test

def create_dataset(dataset, look_back=1):
    '''
    Input: dataset and the number of days to look back (lags).
    
    Function creates lists for X and y. The function iterates through the 
    dataset and appends the lists such that X = t and y = t + look_back
    
    Returns: numpy arrays for X and y
    '''
    
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

def fit_model(series_train, pdq=(1,0,1),pdqs=(0,0,0,1)):
    '''
    Input: a training and test set that are in Series format, the order, and 
    seasonal order for Time Series modeling
    
    Function compiles the model and fits it to the training set. Then prints 
    the model summary and plots the diagnostic report.
    
    Returns: fit model results
    '''
    
    model = sm.tsa.statespace.SARIMAX(series_train,order=pdq,seasonal_order=pdqs)
    results = model.fit()
    results.summary
    residuals = results.resid
    print(results.summary())
    results.plot_diagnostics(figsize=(11,8))
    plt.show();
    return results


def test_RMSE(series_train, series_test, pdq, pdqs, display=True):
    '''
    Input: training and test set, the order, and seasonal order for ARIMA 
    modeling.  
    
    Creates list of predictions by fitting the model to the training values, 
    creating a forecast and recording the predicted values.  Making use of 
    Scikit Learn's MSE function, the test and prediction values are passed 
    through as arguements and then the function prints the square root of the 
    result.
    
    Outputs: RMSE for model and plot of Test predictions against actual values 
    from model.
    '''
    
    X_train = series_train.values
    X_test = series_test.values
    train, test = X_train, X_test
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history, order=pdq,seasonal_order=pdqs)
        fit_model = model.fit(disp=0)
        output = fit_model.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test[t])
    rmse = np.sqrt(MSE(test, predictions))
    print('SARIMA model RMSE on test data: %.5f' % rmse)
    if display:
        plt.figure(figsize=(13,6))
        plt.title('Actual Test Data vs. Predictions')
        plt.plot(history[-51:],label='Actual', color='b')
        plt.plot(predictions,label='Predictions',color='r')
        plt.legend(loc='best')
        plt.show()

def train_RMSE(train, results, display = True):
    '''
    Input: training set and results of fitted model.  
    
    Creates list of predictions by use of model_results.predict() function. 
    Making use of Scikit Learn's MSE function, the test and prediction values 
    are passed through as arguements and then the function prints the square 
    root of the product.
    
    Outputs: RMSE for model and plot of train set predictions against actual 
    values from model.
    '''
    
    train_pred = results.predict(-56)
    rmse = np.sqrt(MSE(train[-56:],train_pred))
    print(f'SARIMA model RMSE on train data: %.5f' % rmse)
    if display:
        plt.figure(figsize=(13,6))
        plt.plot(train[:], label='Actual',color='b')
        plt.plot(train_pred, label='Predicted',color='r')
        plt.legend(loc='best')
        plt.title('Actual Train Data vs. Predictions')
        plt.show();


def LSTM_obtain_error(model, X_train, X_test, y_train, y_test):
    '''
    Input: fitted LSTM model
    
    Function creates prediction on 'X_train' and 'X_test' and inverts the 
    MinMax scaling so that the values are on the correct scale.
    
    Lastly, the function prints out the Train and Test Mean Absolute Errors as 
    well as their Root Mean Squared Errors.
    
    Returns Train
    '''
    
    scaler = MinMaxScaler(feature_range=(0,1))
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions to original scale for interpretation
    train_predict = scaler.inverse_transform(train_predict)
    y_train_rescaled = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_rescaled = scaler.inverse_transform([y_test])

    print('Train Mean Absolute Error:', MAE(y_train_rescaled[0], 
                                            train_predict[:,0]))

    print('Train Root Mean Squared Error:',np.sqrt(MSE(y_train_rescaled[0], 
                                                   train_predict[:,0])))
    print('Test Mean Absolute Error:', MAE(y_test_rescaled[0], 
                                           test_predict[:,0]))

    print('Test Root Mean Squared Error:',np.sqrt(MSE(y_test_rescaled[0], 
                                                  test_predict[:,0])))
    
    
def plot_loss(history):
    '''
    Input: historical object for neural network
    Output: plot of the model loss.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show();
    
    
    
def LSTM_prediction_plot(test_predict, y_test_rescaled): 
    '''
    Input: LSTM model
    
    Function inverts the MinMax scaling of the various training and test sets, 
    and plots the predictions.
    
    Output: Plot
    '''
    aa=[x for x in range(11)]
    plt.figure(figsize=(8,4))
    plt.plot(aa, y_test_rescaled[0][:11], marker='.', label="actual")
    plt.plot(aa, test_predict[:,0][1:12], '-', label="prediction")
    # plt.tick_params(left=False, labelleft=True) #remove ticks
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Increase in COVID-19 Cases', size=15)
    plt.xlabel('Time step', size=15)
    plt.title('Predicted Increase in Covid Cases vs. Actual Data')
    plt.legend(fontsize=15)
    plt.show();