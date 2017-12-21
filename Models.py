import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import os
import pickle
from sklearn.model_selection import TimeSeriesSplit
import Visualiztion as vs

fullList = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA3', 'EMA6', 'EMA12', 'EMA26', 'MACD12_26_9',
            'MACD_signal12_26_9', 'SMA14', 'Upper_band14', 'Lower band14', 'RSI6EMA', 'RSI12EMA', 'Momentum1', 'Momentum3',
            'RateOfChange3', 'RateOfChange12', 'CCI12', 'CCI20', 'WILLR14', 'ATR14', 'TripleEMA6', 'OBV', 'MFI14', 'PDI14',
            'NDI14', 'ADX14', 'PDI20', 'NDI20', 'ADX20', 'UpDown1', 'UpDown3', 'UpDown5', 'UpDown7', 'UpDown10', 'UpDown15',
            'UpDown30', 'Open_SP500', 'High_SP500', 'Low_SP500', 'Close_SP500', 'Adj Close_SP500', 'Volume_SP500', 'SMA3_SP500',
            'EMA6_SP500', 'EMA12_SP500', 'EMA26_SP500', 'MACD12_26_9_SP500', 'MACD_signal12_26_9_SP500', 'SMA14_SP500',
            'Upper_band14_SP500', 'Lower band14_SP500', 'RSI6EMA_SP500', 'RSI12EMA_SP500', 'Momentum1_SP500',
            'Momentum3_SP500', 'RateOfChange3_SP500', 'RateOfChange12_SP500', 'CCI12_SP500', 'CCI20_SP500',
            'WILLR14_SP500', 'ATR14_SP500', 'TripleEMA6_SP500', 'OBV_SP500', 'MFI14_SP500', 'PDI14_SP500',
            'NDI14_SP500', 'ADX14_SP500', 'PDI20_SP500', 'NDI20_SP500', 'ADX20_SP500', 'UpDown1_SP500',
            'UpDown3_SP500', 'UpDown5_SP500', 'UpDown7_SP500', 'UpDown10_SP500', 'UpDown15_SP500',
            'UpDown30_SP500', 'Open_NASDAQ', 'High_NASDAQ', 'Low_NASDAQ', 'Close_NASDAQ', 'Adj Close_NASDAQ',
            'Volume_NASDAQ', 'SMA3_NASDAQ', 'EMA6_NASDAQ', 'EMA12_NASDAQ', 'EMA26_NASDAQ', 'MACD12_26_9_NASDAQ',
            'MACD_signal12_26_9_NASDAQ', 'SMA14_NASDAQ', 'Upper_band14_NASDAQ', 'Lower band14_NASDAQ', 'RSI6EMA_NASDAQ',
            'RSI12EMA_NASDAQ', 'Momentum1_NASDAQ', 'Momentum3_NASDAQ', 'RateOfChange3_NASDAQ', 'RateOfChange12_NASDAQ',
            'CCI12_NASDAQ', 'CCI20_NASDAQ', 'WILLR14_NASDAQ', 'ATR14_NASDAQ', 'TripleEMA6_NASDAQ', 'OBV_NASDAQ', 'MFI14_NASDAQ',
            'PDI14_NASDAQ', 'NDI14_NASDAQ', 'ADX14_NASDAQ', 'PDI20_NASDAQ', 'NDI20_NASDAQ', 'ADX20_NASDAQ', 'UpDown1_NASDAQ',
            'UpDown3_NASDAQ', 'UpDown5_NASDAQ', 'UpDown7_NASDAQ', 'UpDown10_NASDAQ', 'UpDown15_NASDAQ', 'UpDown30_NASDAQ']

featuresList1 = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA3', 'EMA6', 'EMA12', 'EMA26',
           'MACD12_26_9', 'MACD_signal12_26_9', 'SMA14', 'Upper_band14', 'Lower band14', 'RSI6EMA', 'RSI12EMA',
            'Momentum1', 'Momentum3', 'RateOfChange3', 'RateOfChange12', 'CCI12', 'CCI20', 'WILLR14', 'ATR14',
           'TripleEMA6', 'OBV', 'MFI14', 'PDI14', 'NDI14', 'ADX14', 'PDI20', 'NDI20', 'ADX20']

featuresList3 = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA3', 'EMA6', 'EMA12', 'EMA26', 'MACD12_26_9',
            'MACD_signal12_26_9', 'SMA14', 'Upper_band14', 'Lower band14', 'RSI6EMA', 'RSI12EMA', 'Momentum1', 'Momentum3',
            'RateOfChange3', 'RateOfChange12', 'CCI12', 'CCI20', 'WILLR14', 'ATR14', 'TripleEMA6', 'OBV', 'MFI14', 'PDI14',
            'NDI14', 'ADX14', 'PDI20', 'NDI20', 'ADX20', 'Open_SP500', 'High_SP500', 'Low_SP500', 'Close_SP500', 'Adj Close_SP500', 'Volume_SP500', 'SMA3_SP500',
            'EMA6_SP500', 'EMA12_SP500', 'EMA26_SP500', 'MACD12_26_9_SP500', 'MACD_signal12_26_9_SP500', 'SMA14_SP500',
            'Upper_band14_SP500', 'Lower band14_SP500', 'RSI6EMA_SP500', 'RSI12EMA_SP500', 'Momentum1_SP500',
            'Momentum3_SP500', 'RateOfChange3_SP500', 'RateOfChange12_SP500', 'CCI12_SP500', 'CCI20_SP500',
            'WILLR14_SP500', 'ATR14_SP500', 'TripleEMA6_SP500', 'OBV_SP500', 'MFI14_SP500', 'PDI14_SP500',
            'NDI14_SP500', 'ADX14_SP500', 'PDI20_SP500', 'NDI20_SP500', 'ADX20_SP500', 'Open_NASDAQ', 'High_NASDAQ', 'Low_NASDAQ', 'Close_NASDAQ', 'Adj Close_NASDAQ',
            'Volume_NASDAQ', 'SMA3_NASDAQ', 'EMA6_NASDAQ', 'EMA12_NASDAQ', 'EMA26_NASDAQ', 'MACD12_26_9_NASDAQ',
            'MACD_signal12_26_9_NASDAQ', 'SMA14_NASDAQ', 'Upper_band14_NASDAQ', 'Lower band14_NASDAQ', 'RSI6EMA_NASDAQ',
            'RSI12EMA_NASDAQ', 'Momentum1_NASDAQ', 'Momentum3_NASDAQ', 'RateOfChange3_NASDAQ', 'RateOfChange12_NASDAQ',
            'CCI12_NASDAQ', 'CCI20_NASDAQ', 'WILLR14_NASDAQ', 'ATR14_NASDAQ', 'TripleEMA6_NASDAQ', 'OBV_NASDAQ', 'MFI14_NASDAQ',
            'PDI14_NASDAQ', 'NDI14_NASDAQ', 'ADX14_NASDAQ', 'PDI20_NASDAQ', 'NDI20_NASDAQ', 'ADX20_NASDAQ']

dropList1 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Open_SP500', 'High_SP500', 'Low_SP500', 'Close_SP500', 'Volume_SP500', 'Open_NASDAQ', 'High_NASDAQ', 'Low_NASDAQ', 'Close_NASDAQ',
            'Volume_NASDAQ']

allLabelsList = ['UpDown1', 'UpDown3', 'UpDown5', 'UpDown7', 'UpDown10', 'UpDown15', 'UpDown30']

companyList = ['MMM', 'NFLX', 'AAPL', 'XOM', 'BAC']

def getStock(ticker):
    path = 'stockIndicator_dfs\{}.csv'.format(ticker)
    if not os.path.exists(path):
        print('{} not exist'.format(ticker))
        return None
    df = pd.read_csv(
            path.format(ticker),            
            index_col = 0,
            parse_dates = True
        )
    return df

def combineDf(ticker):
    df = getStock(ticker)
    dfSP500 = getStock('SP500')
    dfNASDAQ = getStock('NASDAQ')
    dfSP500 = dfSP500.add_suffix('_SP500')
    dfNASDAQ = dfNASDAQ.add_suffix('_NASDAQ')
    df = df.join(dfSP500, how = 'outer')
    df = df.join(dfNASDAQ, how = 'outer')
    return df

def addBenchmark(df, days = 10):
    if not 'SMA{}'.format(days) in df.columns:
        df['SMA{}'.format(days)] = df['Adj Close'].rolling(window = days).mean()
    diff = df['Adj Close'] - df['SMA{}'.format(days)]
    df['Benchmark{}'.format(days)] = diff.map(lambda x: 1 if x > 0 else -1)
    return df

def getXy(ticker, startTrainDate = dt.date(2010,1,1),
          endTrainDate = dt.date(2015,12,31),
          startTestDate = dt.date(2016,1,1), endTestDate = dt.date(2016,12,31),
          predict_period = 10, featuresList = featuresList3,
          isDrop = False, dropList = dropList1, isNormalized = True):
    df = combineDf(ticker)
    upDown = df['UpDown{}'.format(predict_period)]    

    features_raw = df[featuresList]

    if(isDrop == True):
        features_raw = features_raw.drop(dropList, axis=1)
        
    features_raw = features_raw.ix[startTrainDate:endTestDate]
    X_train = features_raw.ix[startTrainDate:endTrainDate]
    y_train = upDown.ix[startTrainDate:endTrainDate]
    X_test = features_raw.ix[startTestDate:endTestDate]
    y_test = upDown.ix[startTestDate:endTestDate]
    
    l1 = features_raw.columns.tolist()
    if isNormalized:
        scaler = MinMaxScaler()
        features_transform = pd.DataFrame(data = features_raw)
        features_transform[l1] = scaler.fit_transform(features_transform[l1])
        X_train = features_transform.ix[startTrainDate:endTrainDate]
        X_test = features_transform.ix[startTestDate:endTestDate]


    return X_train, y_train, X_test, y_test


def AdaBoosting(X_train, y_train, isGridSearch = False, parm = {'n_estimators':[50,100,200,400], 'learning_rate':[1,2,3,4]}):
    clf = AdaBoostClassifier()

    if isGridSearch:
        parameters = parm 
        scorer = make_scorer(fbeta_score, beta = 1)
        grid_obj = GridSearchCV(clf, parameters, scoring = scorer)
        grid_fit = grid_obj.fit(X_train, y_train)
        best_clf = grid_fit.best_estimator_
        return ['AdaBoosting', best_clf]
    else:
        return ['AdaBoosting', clf.fit(X_train, y_train)]

def SVM(X_train, y_train, isGridSearch = False, parm = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]):
    clf = svm.SVC(gamma = 0.001)
    if isGridSearch:
        my_cv = TimeSeriesSplit(n_splits=5).split(X_train)
        parameters = parm
        scorer = make_scorer(fbeta_score, beta = 1)
        grid_obj = GridSearchCV(clf, parameters, scoring = scorer, cv = my_cv)
        grid_fit = grid_obj.fit(X_train, y_train)
        #print(grid_fit.best_params_ )
        best_clf = grid_fit.best_estimator_
        return['SVM', best_clf]
    
    clf.fit(X_train, y_train)
    return ['SVM', clf]

def LR(X_train, y_train):
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    return clf

def KNC(X_train, y_train):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    return clf

def NN(X_train, y_train):
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    return clf

def ml(classifier, ticker,
       X_train, y_train,
       X_test, y_test):
    [clf_Name, clf] = classifier(X_train, y_train, isGridSearch = False)

    predictions = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    fscore = fbeta_score(y_test, predictions, beta = 1)
    
    return [ticker, clf_Name, accuracy, fscore]
    
def plotpriceLabel():
    for ticker in companyList:
        X_train, y_train, X_test, y_test = getXy(ticker, isDrop = True)
    
        print('stock: {} accuracy:{} f1score: {}'.format(ticker, accuracy, fscore))
        l1.append([ticker, clf, accuracy, fscore])

    performance_df = pd.DataFrame(l1, columns = ['Ticker', 'Classify', 'Accuracy', 'Fscore'])

    print(performance_df)
    vs.closeLabel(df)

    
def test():
    df = getStock('BAC')
    df = addBenchmark(df)
    #df = df[['SMA10','Adj Close', 'Benchmark10']]
    startTestDate = dt.date(2016,1,1)
    endTestDate = dt.date(2016,12,31)
    predictions = df['Benchmark10'].ix[startTestDate:endTestDate]
    y_test = df['UpDown10'].ix[startTestDate:endTestDate]

    accuracy = accuracy_score(y_test, predictions)
    fscore = fbeta_score(y_test, predictions, beta = 1)

    print('accuracy: {} f1score: {}'.format(accuracy, fscore))

def benchmark(tickers):
    for ticker in tickers:
        df = getStock(ticker)
        df = addBenchmark(df)
        startTestDate = dt.date(2016,1,1)
        endTestDate = dt.date(2016,12,31)
        predictions = df['Benchmark10'].ix[startTestDate:endTestDate]
        y_test = df['UpDown10'].ix[startTestDate:endTestDate]

        accuracy = accuracy_score(y_test, predictions)
        fscore = fbeta_score(y_test, predictions, beta = 1)

        print('stock: {} accuracy: {} f1score: {}'.format(ticker, accuracy, fscore))

def run():

    l1 = []
    
    for ticker in companyList:
        X_train, y_train, X_test, y_test = getXy(ticker, isDrop = True)       
        [ticker, clf, accuracy, fscore] = ml(SVM, ticker, X_train, y_train, X_test, y_test)
        print('stock: {} accuracy:{} f1score: {}'.format(ticker, accuracy, fscore))
        l1.append([ticker, clf, accuracy, fscore])

    performance_df = pd.DataFrame(l1, columns = ['Ticker', 'Classify', 'Accuracy', 'Fscore'])

    print(performance_df)

def timeSeries(ticker = 'AAPL'):
    X_train, y_train, X_test, y_test = getXy(ticker, isDrop = True)
##    X_train1 = X_train.reset_index()
##    y_train1 = y_train.reset_index()
##    groups = X_train1.groupby(X_train1.Date.dt.year).groups
##    
##    sorted_groups = [value for (key, value) in sorted(groups.items())]
##
##    cv = [(sorted_groups[i].tolist() + sorted_groups[i+1].tolist(), sorted_groups[i+2].tolist())
##          for i in range(len(sorted_groups)-2)]
##    print(len(cv))
    my_cv = TimeSeriesSplit(n_splits=5).split(X_train)
    print(my_cv)
    parm = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
         ]
    clf = svm.SVC() 
    scorer = make_scorer(fbeta_score, beta = 1)
    grid_obj = GridSearchCV(clf, parm, scoring = scorer, cv = my_cv)
    grid_fit = grid_obj.fit(X_train, y_train)


    
#timeSeries()   
run()
#benchmark(companyList)

