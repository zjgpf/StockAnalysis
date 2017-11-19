import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier

def prepare1():
    df = pd.read_csv('Stock_Features_APPL_SP500_NASDAQ.csv',
                     index_col = 0,
                     parse_dates = True)
##    droplist = ['diff','diff_NASDAQ','diff_SP500']
##    df = df.drop(droplist, axis = 1)
##    df.to_csv('Stock_Features_APPL_SP500_NASDAQ.csv')

    columns = df.columns.tolist()
    

    #UpDown10: 38
##    print(columns.index('UpDown10'))
    list_f1 = columns[:39]
    df1 = df[list_f1].ix[dt.date(2010,1,1):dt.date(2016,12,31)]
    df1.to_csv('Stock_Features_APPL_2010_2016.csv')
    #print(df1)

def prepare2():
    df = pd.read_csv('Stock_Features_APPL_SP500_NASDAQ.csv',
                 index_col = 0,
                 parse_dates = True)
    columns = df.columns.tolist()
    list_f1 = columns[:39]
    df1 = df[list_f1].ix[dt.date(2001,1,1):dt.date(2016,12,31)]
    df1.to_csv('Stock_Features_APPL_2001_2016.csv')

def prepare3():
    df = pd.read_csv('Stock_Features_APPL_SP500_NASDAQ.csv',
             index_col = 0,
             parse_dates = True)
    columns = df.columns.tolist()
    print(columns)
   # list_f1 = columns[:39]
   # df1 = df[list_f1].ix[dt.date(2001,1,1):dt.date(2016,12,31)]
    #df1.to_csv('Stock_Features_APPL_2001_2016.csv')
    

def testFeatures():
    alllist = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA3', 'EMA6', 'EMA12', 'EMA26', 'MACD12_26_9', 'MACD_signal12_26_9', 'SMA14', 'Upper_band14', 'Lower band14', 'RSI6EMA', 'RSI12EMA',
                'Momentum1', 'Momentum3', 'RateOfChange3', 'RateOfChange12', 'CCI12', 'CCI20', 'WILLR14', 'ATR14', 'TripleEMA6', 'OBV', 'MFI14', 'PDI14', 'NDI14', 'ADX14', 'PDI20', 'NDI20', 'ADX20']
    return ['Adj Close', 'SMA3', 'EMA6','MACD12_26_9',
            'MACD_signal12_26_9', 'SMA14', 'Upper_band14', 'Lower band14']

def getXy(days):
    df = pd.read_csv('Stock_Features_APPL_2010_2016.csv',
                     index_col = 0,
                     parse_dates = True)

    upDown = df['UpDown{}'.format(days)]
    

    features_raw = df.drop(['UpDown1','UpDown3','UpDown5','UpDown7','UpDown10'], axis=1)
    l1 = features_raw.columns.tolist()
    
    scaler = MinMaxScaler()
    features_transform = pd.DataFrame(data = features_raw)
    features_transform[l1] = scaler.fit_transform(features_transform[l1])

    #l2 = [14,16,24,28]
    
    
    #features_transform = features_transform[[features_transform.columns[i] for i in l2]]
    
    X_train = features_transform.ix[dt.date(2001,1,1):dt.date(2015,12,31)]
    y_train = upDown.ix[dt.date(2001,1,1):dt.date(2015,12,31)]
    X_test = features_transform.ix[dt.date(2016,1,1):dt.date(2016,12,31)]
    y_test = upDown.ix[dt.date(2016,1,1):dt.date(2016,12,31)]
    return X_train, y_train, X_test, y_test

def getXy3(days):
    df = pd.read_csv('Stock_Features_APPL_SP500_NASDAQ.csv',
                     index_col = 0,
                     parse_dates = True)
    df = df.ix[dt.date(2001,1,1):]
    upDown = df['UpDown{}'.format(days)]
    dropList = ['UpDown1','UpDown3','UpDown5','UpDown7','UpDown10',
                'UpDown1_SP500','UpDown3_SP500','UpDown5_SP500','UpDown7_SP500','UpDown10_SP500',
                'UpDown1_NASDAQ','UpDown3_NASDAQ','UpDown5_NASDAQ','UpDown7_NASDAQ','UpDown10_NASDAQ']
    features_raw = df.drop(dropList, axis=1)

    l1 = features_raw.columns.tolist()
    
    scaler = MinMaxScaler()
    features_transform = pd.DataFrame(data = features_raw)
    features_transform[l1] = scaler.fit_transform(features_transform[l1])

    X_train = features_transform.ix[dt.date(2010,1,1):dt.date(2015,12,31)]
    y_train = upDown.ix[dt.date(2010,1,1):dt.date(2015,12,31)]
    X_test = features_transform.ix[dt.date(2016,1,1):dt.date(2016,12,31)]
    y_test = upDown.ix[dt.date(2016,1,1):dt.date(2016,12,31)]
    return X_train, y_train, X_test, y_test

def AdaBoosting(X_train, y_train):
    clf = AdaBoostClassifier()

    # TODO: Create the parameters list you wish to tune, using a dictionary if needed.
    # HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
    parameters = {'n_estimators':[50,100,200,400], 'learning_rate':[1,2,3,4]}

    # TODO: Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score, beta = 0.5)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf, parameters, scoring = scorer)

    # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    return best_clf

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

def ml():

    X_train, y_train, X_test, y_test = getXy3(10)

##    clf = NN(X_train, y_train)
##    accuracy = clf.score(X_test, y_test)
##
##    print(accuracy)

##    model = AdaBoostClassifier(random_state=0).fit(X_train, y_train)
##    importances = model.feature_importances_
##    print(importances)
##    print(np.where(importances==0.06)[0])


    best_clf = AdaBoosting(X_train, y_train)

    best_predictions = best_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, best_predictions)
    print('Accuracy:{}'.format(accuracy))
    fscore = fbeta_score(y_test, best_predictions, beta = 0.5)
    print('fbeta_score:{}'.format(fscore))
##
ml()
