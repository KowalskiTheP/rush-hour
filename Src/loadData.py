import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=150)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#==============================================================================

def load_fromCSV(csvFile, decimal , seperator, header, dateColumn):
  df=pd.read_csv(csvFile, decimal=decimal ,sep=seperator, header=header)

  if dateColumn != 'None':
    refdate = '01.01.1960'
    date_format = "%d.%m.%Y"
    date_list = []
    b = datetime.strptime(refdate, date_format)
    for i in range(len(df)):
      a = datetime.strptime(df.loc[i,dateColumn], date_format)
      date_list.append(str(a-b))
      date_list[-1] = [int(s) for s in date_list[-1].split() if s.isdigit() ]
    date_list = np.array(date_list).flatten()
    df.days = date_list   
  return df

#==============================================================================

def getDataSet(dataframe, columns, trainTestSplit):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]

  lastXpercent = int(np.floor(len(dataframe)*trainTestSplit))
  firstXpercent = len(dataframe) - lastXpercent
  trainSet = dataSet[0:firstXpercent]
  testSet = dataSet[firstXpercent:firstXpercent+lastXpercent]
  return trainSet, testSet

#============================================================================== !!!!

def split_data(dataSet, trainTestSplit):
  lastXpercent = int(np.floor(len(dataSet)*trainTestSplit))
  firstXpercent = len(dataSet) - lastXpercent
  trainSet = dataSet[0:firstXpercent]
  testSet = dataSet[firstXpercent:firstXpercent+lastXpercent]
  return trainSet, testSet

#============================================================================== !!!

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n >= maxn:
        return maxn
    else:
        return n
    
#============================================================================== !!!!

def smoothing(data, config):
  data_tmp = data.copy()
  data_return = data.copy()
  smooth = int(config.smoothingparam)
  yColumn = int(config.y_column)
  for i in range(smooth+1,len(data)):
    smoothSum = 0.
    w = 1.
    for j in range(smooth):
      m = abs(data_tmp[i-j,yColumn]-data_tmp[i-j-1,yColumn])
      if i-20 > 0:
        std = np.std(data_tmp[i-20:i,yColumn])
      else:
        std = np.std(data_tmp[i:i+20,yColumn])
      wj = clamp((m/std),(1./float(smooth)),1.)
      if w - wj <= 0.:
        wj = w
        w = 0.
      else:
        w = w - wj
      smoothSum = smoothSum + data_tmp[i-j,yColumn] * wj
    data_return[i,yColumn] = smoothSum
  return data_return

#==============================================================================  !!!!

def getDataSet_noSplit(dataframe, columns):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  return dataSet

#============================================================================== !!!!

def normalise_data_refValue(refValue,data):
  normData = []
  for i in range(len(data)):
    normData.append( data[i]/refValue )
  return np.array(normData)

#============================================================================== !!!!
  
def denormalise_data_refValue(refValue,normData):
  denormData = []
  for i in range(len(normData)):
    denormData.append( refValue * normData[i] )
  return np.array(denormData)

#============================================================================== !!!!

def get_windows_andShift_seq_hourly(x,winLength,look_back,outDim,y_column):
  x_train, y_train = [], []
  #Porbably important bugfix
  #Old: for i in xrange(0,len(x)-(winLength+outDim),2): 
  for i in xrange(0,len(x)-(winLength+look_back+outDim),1):
    x_train.append(x[i:i+winLength])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),y_column])
  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

#============================================================================== !!!! 

def minMaxNorm(xWinTrain, yWinTrain, yColumn,yNorm=True):
  trainMin, trainMax = [], []
  for j in range(len(xWinTrain)):
    tmpMin, tmpMax = [], []
    for i in range(len(xWinTrain[0,0])):
      tmpMin.append(np.amin(xWinTrain[j,:,i]))
    xWinTrain[j,:] = xWinTrain[j,:] - tmpMin
    if yNorm==True:
      yWinTrain[j] = yWinTrain[j] - tmpMin[yColumn]
    for i in range(len(xWinTrain[0,0])):
      tmpMax.append(np.amax(xWinTrain[j,:,i]))
    trainMin.append(tmpMin)
    trainMax.append(tmpMax)
  return np.array(trainMin), np.array(trainMax), xWinTrain, yWinTrain

#==============================================================================

def make_windowed_data_withSplit(dataframe, config):

  refValue = float(config.refvalue)
  winL = int(config.winlength)
  lookB = int(config.look_back)
  xDim = len(config.columns)
  yLen = int(config.outputlength)
  y_column = int(config.y_column)
  dataSet_Full = getDataSet_noSplit(dataframe, config.columns)
  dataSet_Full_tmp = dataSet_Full.copy()

  #dataSetTrain, dataSetTest = split_data(dataSet_Full, float(config.traintestsplit))

#=========================================================================================
# The russian data is very dirty! If its cleaned up, only a few hundred days are left. Therefore its probably the best, if we use it in its dirty original form. 
#The following code was used to clean the data up.

  #tmp, xdata = [], []
  #stop = 0
  #for i in range(1,len(dataSet_Full)): 
    #if dataSet_Full[i,0] == dataSet_Full[i-1,0] and dataSet_Full[i,2] != dataSet_Full[i-1,2]:
      #if dataSet_Full[i,1] >= 7. and dataSet_Full[i,1] < 17.:
        #tmp.append(dataSet_Full[i])
    #else:
      #if len(tmp)==8:
        #xdata.append(tmp)
      #tmp = []
      #if dataSet_Full[i,1] >= 7. and dataSet_Full[i,1] < 17.:
        #tmp.append(dataSet_Full[i])
  #xdata = np.array(xdata)
  #xdata = np.reshape(xdata,(len(xdata)*8,xDim))
  #dataSet_Full = np.array(xdata)
  
#=========================================================================================

  dataSetTrain, dataSetTest = split_data(dataSet_Full, config.traintestsplit)
  
  if config.smoothingswitch == 'on':
    dataSetTrain = smoothing(dataSetTrain, config)
    dataSetTest_smooth = smoothing(dataSetTest, config)
  
  x_winTrain, y_winTrain = get_windows_andShift_seq_hourly(dataSetTrain, winL, lookB,yLen,y_column)
  x_winTest, y_winTest = get_windows_andShift_seq_hourly(dataSetTest, winL, lookB,yLen,y_column)
  
  print 'x_winTrain',x_winTrain[0]
  print 'y_winTrain',y_winTrain[0]
  
  if config.smoothingswitch == 'on':
    x_winTest_smooth, y_winTest_smooth = get_windows_andShift_seq_hourly(dataSetTest_smooth, winL, lookB,yLen,y_column)
    x_winTest = x_winTest_smooth

  # Deleting evening to morning predictions
  if 0 == 1:
    x_tmp, y_tmp = [],[]
    for i in range(len(x_winTrain)):
      #if x_winTrain[i,-1,1] != 16 and x_winTrain[i,-1,1] != 17:
      #if x_winTrain[i,0,1] == 7. or x_winTrain[i,0,1] == 8. or x_winTrain[i,0,1] == 9.:
      if x_winTrain[i,0,1] == 8. or x_winTrain[i,0,1] == 9.:
        x_tmp.append(x_winTrain[i])
        y_tmp.append(y_winTrain[i])
    x_winTrain = np.array(x_tmp)
    y_winTrain = np.array(y_tmp)

    x_tmp, y_tmp = [],[]
    for i in range(len(x_winTest)):
      #if x_winTest[i,-1,1] != 16 and x_winTest[i,-1,1] != 17:
      #if x_winTest[i,0,1] == 7. or x_winTest[i,0,1] == 8. or x_winTest[i,0,1] == 9.:
      if x_winTest[i,0,1] == 8. or x_winTest[i,0,1] == 9.:
        x_tmp.append(x_winTest[i])
        y_tmp.append(y_winTest[i])
    x_winTest = np.array(x_tmp)
    y_winTest = np.array(y_tmp)

#=====================  
  if 0 == 1:
    x_tmp, y_tmp = [],[]
    for i in range(len(x_winTrain)):
      if x_winTrain[i,-1,0] != 6.:
        x_tmp.append(x_winTrain[i])
        y_tmp.append(y_winTrain[i])
    x_winTrain = np.array(x_tmp)
    y_winTrain = np.array(y_tmp)

    x_tmp, y_tmp = [],[]
    for i in range(len(x_winTest)):
      if x_winTest[i,-1,0] != 6.:
        x_tmp.append(x_winTest[i])
        y_tmp.append(y_winTest[i])
    x_winTest = np.array(x_tmp)
    y_winTest = np.array(y_tmp)  

  if config.normalise == 3:
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm,trainRef,testRef = [],[],[],[],[],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append(normalise_data_refValue(config.refvalue,x_winTrain[i]))
      y_winTrain_norm.append(normalise_data_refValue(config.refvalue,y_winTrain[i]))
      trainRef.append(x_winTrain[i,-1])
    for i in range(len(y_winTest)):
      x_winTest_norm.append( normalise_data_refValue(config.refvalue,x_winTest[i]))
      y_winTest_norm.append( normalise_data_refValue(config.refvalue,y_winTest[i]))
      testRef.append(x_winTest[i,-1])

  if config.normalise == 4:
    trainMin,trainMax,x_winTrain,y_winTrain = minMaxNorm(x_winTrain,y_winTrain,y_column,yNorm=True)
    testMin ,testMax ,x_winTest ,y_winTest  = minMaxNorm(x_winTest ,y_winTest ,y_column,yNorm=True)
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm = [],[],[],[]
    for i in range(len(trainMax)):
      for j in range(len(trainMax[i])):
        if trainMax[i,j] == 0.:
          trainMax[i,j] = 1.
    for i in range(len(testMax)):
      for j in range(len(testMax[i])):
        if testMax[i,j] == 0.:
          testMax[i,j] = 1.
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append( normalise_data_refValue(trainMax[i]         ,x_winTrain[i]) )
      y_winTrain_norm.append( normalise_data_refValue(trainMax[i,y_column],y_winTrain[i]) )
    for j in range(len(y_winTest)):
      x_winTest_norm.append(  normalise_data_refValue(testMax[j]          ,x_winTest[j])  )
      y_winTest_norm.append(  normalise_data_refValue(testMax[j,y_column] ,y_winTest[j])  )

  if config.normalise != 0:
    x_winTrain_norm = np.reshape(np.array(x_winTrain_norm),(len(x_winTrain_norm),winL,xDim ))
    y_winTrain_norm = np.reshape(np.array(y_winTrain_norm),(len(y_winTrain_norm),yLen ))
    x_winTest_norm =  np.reshape(np.array(x_winTest_norm) ,(len(x_winTest_norm) ,winL,xDim ))
    y_winTest_norm =  np.reshape(np.array(y_winTest_norm) ,(len(y_winTest_norm) ,yLen ))

  if config.normalise == 0:
    return x_winTrain, y_winTrain, x_winTest, y_winTest

  if config.normalise == 3:
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainRef), np.array(testRef)
  
  if config.normalise == 4:
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainMax),np.array(trainMin),np.array(testMax),np.array(testMin)  
  
#=========================================================================================

  
