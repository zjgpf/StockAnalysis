import pandas as pd
import numpy as np
import datetime as dt

ticker = 'AAPL'
def getStock(ticker):
    df = pd.read_csv(
            '.\stock_dfs\{}.csv'.format(ticker),            
            index_col = 0,
            parse_dates = True
        )
    return df

def SMA(df, days):
    if not 'SMA{}' in df.columns:
        df['SMA{}'.format(days)] = df['Adj Close'].rolling(window = days, min_periods = 0).mean()
    return df
    

def EMA(df, days):
    if not 'EMA{}' in df.columns:
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

##Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
##Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)-
##Price(t-1)>0};
##PriceDown(t)=1*(Price(t-1)-Price(t)){Price(t)-
##Price(t-1)<0};
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

##If the closing price is above the prior close price then: 
##Current OBV = Previous OBV + Current Volume
##
##If the closing price is below the prior close price then: 
##Current OBV = Previous OBV  -  Current Volume
##
##If the closing prices equals the prior close price then:
##Current OBV = Previous OBV (no change)
def OBV(df,  startDay):
    volume = df['Volume'].ix[startDay:].values.tolist()
    close = df['Adj Close'].ix[startDay:].values.tolist()
    offset = len(df['Volume']) - len(volume)
    obv = []
    obv.append(volume[0])
    for i in range(1,len(volume)):
        if(close[i]>close[i-1]):
            obv.append(volume[i]+obv[i-1])
        elif(close[i]==close[i-1]):
            obv.append(obv[i-1])
        else:
            obv.append(obv[i-1]-volume[i])
    for i in range(0,offset):
        obv.insert(0,0)
    df['OBV'] = pd.Series(obv).values
    return df
    

##Money Flow Index
##Typical Price = (High + Low + Close)/3
##
##Raw Money Flow = Typical Price x Volume
##Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)
##
##Money Flow Index = 100 - 100/(1 + Money Flow Ratio)
def MFI(df, days = 14):
    TP = (df['High']+df['Low']+df['Close'])/3
    RMF = TP*df['Volume']
    delta = TP -TP.shift(1)
    delta.fillna(0, inplace = True)
    deltaList = delta.values.tolist()
    upDownList = map(lambda x: 1 if x > 0 else ( -1 if x < 0 else 0), deltaList)
    upDown = pd.Series(upDownList).values
    RMF = RMF*upDown
    PMF, NMF = RMF.copy(), RMF.copy()
    PMF[PMF<0]=0
    NMF[NMF>0]=0
    PeriodPMF = PMF.rolling(window = days).sum()
    PeriodNMF = NMF.rolling(window = days).sum()
    MFR = abs(PeriodPMF/PeriodNMF)
    MFI = 100 - 100/(1.0+MFR)
    df['MFI{}'.format(days)] = MFI
    return df

##Calculate the True Range (TR), Plus Directional Movement (+DM) and Minus Directional Movement (-DM) for each period.
##Smooth these periodic values using Wilder's smoothing techniques. These are explained in detail in the next section.
##Divide the 14-day smoothed Plus Directional Movement (+DM) by the 14-day smoothed True Range to find the 14-day Plus Directional Indicator (+DI14).
##Multiply by 100 to move the decimal point two places. This +DI14 is the green Plus Directional Indicator line (+DI) that is plotted along with the ADX line.
##Divide the 14-day smoothed Minus Directional Movement (-DM) by the 14-day smoothed True Range to find the 14-day Minus Directional Indicator (-DI14).
##Multiply by 100 to move the decimal point two places. This -DI14 is the red Minus Directional Indicator line (-DI) that is plotted along with the ADX line.
##The Directional Movement Index (DX) equals the absolute value of +DI14 less -DI14 divided by the sum of +DI14 and -DI14. Multiply the result by 100 to move the decimal point over two places.
##After all these steps, it is time to calculate the Average Directional Index (ADX) line. The first ADX value is simply a 14-day average of DX.
##Subsequent ADX values are smoothed by multiplying the previous 14-day ADX value by 13, adding the most recent DX value, and dividing this total by 14.
def ADX_test(df, days = 14):
    HL = df['High'] - df['Low']
    HC = abs(df['High'] - df['Close'].shift(1))
    LC = abs(df['Low'] - df['Close'].shift(1))
    

    temp_df = pd.DataFrame(index = df.index)
    

    temp_df['HL'] = HL
    temp_df['HC'] = HC
    temp_df['LC'] = LC
    temp_df['TR'] = temp_df.max(axis = 1)
    #temp_df.drop(['HL','HC','LC'], axis=1,inplace=True)
    temp_df['High'] = df['High']
    temp_df['Low'] = df['Low']
    temp_df['Close'] = df['Close']
    temp_df = temp_df[['High','Low','Close','TR']]
    temp_df['PDM'] = df['High']-df['High'].shift(1)
    temp_df['NDM'] = df['Low'] - df['Low'].shift(1)

    PDM = temp_df['PDM']
    NDM = temp_df['NDM']
    PDM[PDM<0]=0
    NDM[NDM<0]=0
    PDM[PDM<NDM]=0
    NDM[NDM<PDM]=0

    TR = temp_df['TR']
    period_TR = TR[1:].rolling(window=days).sum()
    period_PDM = PDM.rolling(window=days).sum()
    period_NDM = NDM.rolling(window=days).sum()
    PDI = (period_PDM/period_TR)*100
    NDI = (period_NDM/period_TR)*100

    Diff = abs(PDI-NDI)
    Sum = PDI+NDI
    DX = Diff/Sum*100
    ADX = DX.rolling(window = days).mean()
    temp_df['TR14']= period_TR
    temp_df['PDM14']= period_PDM
    temp_df['NDM14']= period_NDM
    temp_df['PDI14']=PDI
    temp_df['NDI14']=NDI
    temp_df['Diff'] = Diff
    temp_df['Sum'] = Sum
    temp_df['DX']=DX
    temp_df['ADX']=ADX
    df['PDI{}'.format(days)] = PDI
    df['NDI{}'.format(days)] = NDI
    df['ADX{}'.format(days)] = ADX
    
    return df

def ADX(df, days = 14):
    HL = df['High'] - df['Low']
    HC = abs(df['High'] - df['Close'].shift(1))
    LC = abs(df['Low'] - df['Close'].shift(1))
    

    temp_df = pd.DataFrame(index = df.index)
    

    temp_df['HL'] = HL
    temp_df['HC'] = HC
    temp_df['LC'] = LC
    temp_df['TR'] = temp_df.max(axis = 1)
    temp_df['PDM'] = df['High']-df['High'].shift(1)
    temp_df['NDM'] = df['Low'] - df['Low'].shift(1)

    PDM = temp_df['PDM']
    NDM = temp_df['NDM']
    PDM[PDM<0]=0
    NDM[NDM<0]=0
    PDM[PDM<NDM]=0
    NDM[NDM<PDM]=0

    TR = temp_df['TR']
    period_TR = TR[1:].rolling(window=days).sum()
    period_PDM = PDM.rolling(window=days).sum()
    period_NDM = NDM.rolling(window=days).sum()
    PDI = (period_PDM/period_TR)*100
    NDI = (period_NDM/period_TR)*100

    Diff = abs(PDI-NDI)
    Sum = PDI+NDI
    DX = Diff/Sum*100
    ADX = DX.rolling(window = days).mean()

    df['PDI{}'.format(days)] = PDI
    df['NDI{}'.format(days)] = NDI
    df['ADX{}'.format(days)] = ADX
    
    return df

df = getStock(ticker)
df = ADX(df)
#df.to_csv('adx_v3.csv')
#df = OBV(df, dt.date(2017,1,4))
print(df)
#print(df[['Adj Close','CCI20']])
