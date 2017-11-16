import pandas as pd
import numpy as np

ticker = 'AAPL'
def getStock(ticker):
    df = pd.read_csv(
            '.\stock_dfs\{}.csv'.format(ticker),            
            index_col = 0
        )
    return df

def SMA(df, days):
    df['SMA{}'.format(days)] = df['Adj Close'].rolling(window = days, min_periods = 0).mean()
    return df
    

def EMA(df, days):
    df['EMA{}'.format(days)] = df['Adj Close'].ewm(span=days).mean()
    return df

def MACD(df, fastMA_Period = 12, slowMA_Period = 26, signal_Period = 9):
    EMA(df, fastMA_Period)
    EMA(df, slowMA_Period)
    df['MACD'] = df['EMA{}'.format(fastMA_Period)] - df['EMA{}'.format(slowMA_Period)]
    df['MACD_signal'] = pd.ewma(df['MACD'], span=signal_Period)
    return df

def Bollinger_bands(df, days = 14):
    SMA(df, days)
    std = df['Adj Close'].rolling(window = days, min_periods = 0).std()
    df['Upper_band'] = df['SMA{}'.format(days)] + 2*std
    df['Lower band'] = df['SMA{}'.format(days)] - 2*std
    return df

def RSI(df, days=6, method = 'EMA'):
    close = df['Adj Close']
    delta = close-close.shift(1)
    delta.fillna(0, inplace = True)
    up, down = delta.copy(), delta.copy()
    up[up<0] = 0
    down[down>0] = 0
    roll_up = pd.DataFrame()
    roll_down = pd.DataFrame ()
    
    if(method == 'SMA'):
        roll_up = up.rolling(window=days, min_periods = 0).mean()
        roll_down = down.abs().rolling(window=days, min_periods = 0).mean()

    if(method == 'EMA'):
        roll_up = up.ewm(span=days).mean()
        roll_down = down.abs().ewm(span = days).mean()  
    RS = roll_up/roll_down
    RS.fillna(1, inplace = True)
    
    df['RSI{}{}'.format(days,method)] = 100.0 - (100.0/(1.0 + RS))
    return df

def Momentum(df, days):
    df['Momentum{}'.format(days)] = df['Adj Close'] - df['Adj Close'].shift(days)
    df.fillna(0, inplace = True)
    return df

def RateOfChange(df, days):
    df['RateOfChange{}'.format(days)] = (df['Adj Close']/df['Adj Close'].shift(days))*100
    df.fillna(100, inplace = True)


##CCI = (Typical Price  -  n-period SMA of TP) / (Constant x Mean Deviation)
##
##Typical Price (TP) = (High + Low + Close)/3
##
##Constant = .015
def CCI(df, days = 20):
    TP = (df['High']+df['Low']+df['Close'])/3
    SMATP = TP.rolling(window = days, min_periods = 0).mean()
    STDTP = TP.rolling(window = days, min_periods = 0).std()
    df['CCI{}'.format(days)] = (TP - SMATP)/(0.015*STDTP)
    df.fillna(0, inplace = True)   
    return df

##%R = (Highest High - Close)/(Highest High - Lowest Low) * -100
##
##Lowest Low = lowest low for the look-back period
##Highest High = highest high for the look-back period
##%R is multiplied by -100 correct the inversion and move the decimal.
def WillR(df, days = 14):
    HH = df['High'].rolling(window = days, min_periods = 0).max()
    LL = df['Low'].rolling(window = days, min_periods = 0).min()
    WillR = (HH-df['Close'])/(HH-LL)*(-100)
    df['WillR{}'.format(days)] = WillR
    return df

##ATR(t)=((n-1)*ATR(t-1)+Tr(t))/n where
##Tr(t)=Max(Abs(High-Low), Abs(Hight-Close(t-1)),
##Abs(Low-Close(t-1));
def ATR(df, days = 14):
    length = len(df['Close'])
    HL = df['High'] - df['Low']
    HC = abs(df['High'] - df['Close'].shift(1))
    LC = abs(df['Low'] - df['Close'].shift(1))
    

    temp_df = pd.DataFrame(index = df.index)

    temp_df['HL'] = HL
    temp_df['HC'] = HC
    temp_df['LC'] = LC
    temp_df['TR'] = temp_df.max(axis = 1)
    temp_df['High'] = df['High']
    temp_df['Low'] = df['Low']
    temp_df['Close'] = df['Close']
    temp_df['ATR1'] = temp_df['TR'].rolling(window = days).mean()
    temp_df['ATR'] = pd.Series(np.zeros(length), index = temp_df.index)
    temp_df['ATR'].iloc[days-1:days*2-2]=temp_df['ATR1'].iloc[days-1:days*2-2]
    ATR13 = temp_df['ATR'].iloc[days-1:days*2-2].values
    TR14 = temp_df['TR'][days*2-2:].values
    ATR = pd.Series(ATRCal(ATR13,TR14)).values
    temp_df['ATR'] = ATR

    df['ATR{}'.format(days)] = temp_df['ATR']

    return df

def ATRCal(ATR13, TR14):
    ret = list(ATR13)
    for i in range(0, len(TR14)):
        newATR = (sum(ret[i:i+13])+TR14[i])/14
        ret.append(newATR)
    for i in range(0,13):
        ret.insert(0,0)
    return ret

##TR(t)/TR(t-1) where
##TR(t)=EMA(EMA(EMA(Price(t)))) over n days
##period
##Triple
##Exponential
##Moving
##Average
def TEMA(df, days):
    EMA(df, days)
    temp_df = pd.DataFrame(index = df.index)
    temp_df['DoubleEMA'] = df['EMA{}'.format(days)].ewm(span=days).mean()
    temp_df['TripleEMA'] = temp_df['DoubleEMA'].ewm(span=days).mean()
    temp_df['TripleEMA'] = 3*df['EMA{}'.format(days)]-3*temp_df['DoubleEMA']+temp_df['TripleEMA']
    df['TripleEMA{}'.format(days)] = temp_df['TripleEMA']
    return df

df = getStock(ticker)
df = TEMA(df,10)
print(df)
#print(df[['Adj Close','CCI20']])

