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

def getDataSet_noSplit(dataframe, columns, yInDF):
  dataSet_x = np.zeros((len(dataframe), len(columns)))
  dataSet_y = np.zeros((len(dataframe), len(yInDF)))
  for i in range(len(columns)):
    dataSet_x[:,i] = dataframe.iloc[:,columns[i]]
  for i in range(len(yInDF)):
    dataSet_y[:,i] = dataframe.iloc[:,yInDF[i]]
  return dataSet_x, dataSet_y

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

def get_windows_andShift_seq_hourly(x,y,winLength,look_back,outDim):
  x_train, y_train = [], []
  #Porbably important bugfix
  #Old: for i in xrange(0,len(x)-(winLength+outDim),2): 
  for i in xrange(0,len(x)-(winLength+look_back+outDim),1):
    x_train.append(x[i:i+winLength])
    y_train.append(y[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1)])
  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

#============================================================================== !!!! 

def minMaxNorm_x(xWinTrain):
  trainMin_x, trainMax_x, trainMin_y, trainMax_y = [], [],[], []
  for j in range(len(xWinTrain)):
    tmpMin_x, tmpMax_x, tmpMin_y, tmpMax_y = [], [], [], []
    
    trainMin_x.append(np.amin(xWinTrain[j,:,:]))
    xWinTrain[j,:] = xWinTrain[j,:] - trainMin_x[-1]
    trainMax_x.append(np.amax(xWinTrain[j,:,:]))

    
  return np.array(trainMin_x), np.array(trainMax_x), xWinTrain

#============================================================================== !!!! 

#def minMaxNorm_x(xWinTrain):
  #trainMin_x, trainMax_x, trainMin_y, trainMax_y = [], [],[], []
  #for j in range(len(xWinTrain)):
    #tmpMin_x, tmpMax_x, tmpMin_y, tmpMax_y = [], [], [], []
    #print xWinTrain[0,0]
    #sys.exit()
    #for i in range(len(xWinTrain[0,0])):
      #tmpMin_x.append(np.amin(xWinTrain[j,:,i]))
    #xWinTrain[j,:] = xWinTrain[j,:] - tmpMin_x
    
    
    #for i in range(len(xWinTrain[0,0])):
      #tmpMax_x.append(np.amax(xWinTrain[j,:,i]))
    #trainMin_x.append(tmpMin_x)
    #trainMax_x.append(tmpMax_x)
    
  #return np.array(trainMin_x), np.array(trainMax_x), xWinTrain

#============================================================================== !!!! 

def minMaxNorm_y(y_data,minValue,maxValue):
  min_y, max_y = [], []
  
  for i in range(len(y_data)):
    if minValue == False:
      tmpMin = np.amin(y_data[i,:])
    else:
      tmpMin = minValue
    y_data[i,:] = y_data[i,:] - tmpMin
    
    if maxValue == False:
      tmpMax = np.amax(y_data[i,:])
    else:
      tmpMax = maxValue
      
    min_y.append(tmpMin)
    max_y.append(tmpMax)

  return np.array(min_y), np.array(max_y), y_data

#==============================================================================

def diffOnY(y_data, intervall):
  y_data_new = y_data.copy()
  for i in range(intervall,len(y_data)):
    y_data_new[i] = (y_data[i] - y_data[i-intervall])
  y_data_new[0:intervall] = 0.
  return y_data_new

#============================================================================== !!!! 


def make_windowed_data_withSplit(dataframe, config):
  yNorm=False
  refValue = float(config.refvalue)
  winL = int(config.winlength)
  lookB = int(config.look_back)
  xDim = len(config.columns)
  yLen = int(config.outputlength)
  
  if config.ydiffs=='on':
    dataSet_Full_x, dataSet_Full_y_pre = getDataSet_noSplit(dataframe, config.columns, config.y_column)
    print 'dataSet_Full_y_pre (old):\n', dataSet_Full_y_pre
    print dataSet_Full_y_pre.shape
    dataSet_Full_y = diffOnY(dataSet_Full_y_pre, config.intervall)
    print 'dataSet_Full_y (new):\n', dataSet_Full_y
    print dataSet_Full_y.shape
    print 'ENDE'
  else:
    dataSet_Full_x, dataSet_Full_y = getDataSet_noSplit(dataframe, config.columns, config.y_column)
  
  if config.ydiffs!='on':
    yRefValue=np.mean(dataSet_Full_y)
    yMinValue=np.amin(dataSet_Full_y)
    print 'yRefValue: ', yRefValue
    print 'yMinValue: ', yMinValue  
  
  # This works only if there is just the minute column in the data and it must be on position 0.
  maxTime = np.amax(dataSet_Full_x[:,0])
  dataSet_Full_x[:,0] = dataSet_Full_x[:,0] / maxTime
  
  print dataSet_Full_x
  for j in range(1,xDim):
    #if np.mean(dataSetTrain_x[:,j]) > 100.:
      #dataSetTrain_x[:,j] = dataSetTrain_x[:,j]/4.
    div = np.amin(dataSet_Full_x[:,j])
    dataSet_Full_x[:,j] = dataSet_Full_x[:,j] - div
    print np.mean(dataSet_Full_x[:,j])

  #dataSet_Full_tmp = dataSet_Full.copy()

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

  dataSetTrain_x, dataSetTest_x = split_data(dataSet_Full_x, config.traintestsplit)
  dataSetTrain_y, dataSetTest_y = split_data(dataSet_Full_y, config.traintestsplit)
  
  if config.ydiffs!='on':
    dataSetTrain_y = dataSetTrain_y - yMinValue
    dataSetTest_y  = dataSetTest_y  - yMinValue

  if config.smoothingswitch == 'on':
    dataSetTrain = smoothing(dataSetTrain, config)
    dataSetTest_smooth = smoothing(dataSetTest, config)

  x_winTrain, y_winTrain = get_windows_andShift_seq_hourly(dataSetTrain_x, dataSetTrain_y, winL, lookB,yLen)
  x_winTest, y_winTest   = get_windows_andShift_seq_hourly(dataSetTest_x , dataSetTest_y , winL, lookB,yLen)

  tmp_x_winTrain = []
  tmp_y_winTrain = []
  if config.anticorrelation == 'on':
    for antiCorr in range(len(x_winTrain)):
      if antiCorr % config.overlap == 0:
        tmp_x_winTrain.append(x_winTrain[antiCorr])
        tmp_y_winTrain.append(y_winTrain[antiCorr])
    x_winTrain = np.array(tmp_x_winTrain)
    y_winTrain = np.array(tmp_y_winTrain)

  print 'x_winTrain',x_winTrain[0]
  print 'y_winTrain',y_winTrain[0]
  print 'y_winTrain_min', np.amin(y_winTrain)
  print 'y_winTrain_max', np.amax(y_winTrain)

  if config.smoothingswitch == 'on':
    x_winTest_smooth, y_winTest_smooth = get_windows_andShift_seq_hourly(dataSetTest_smooth, winL, lookB,yLen,config.y_column)
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
      if config.ydiffs!='on':
        y_winTrain_norm.append(normalise_data_refValue(config.refvalue,y_winTrain[i]))
      trainRef.append(config.refvalue)
    for i in range(len(y_winTest)):
      x_winTest_norm.append( normalise_data_refValue(config.refvalue,x_winTest[i]))
      if config.ydiffs!='on':
        y_winTest_norm.append( normalise_data_refValue(config.refvalue,y_winTest[i]))
      testRef.append(config.refvalue)

  if config.normalise == 4:
    trainMin_x, trainMax_x, x_winTrain = minMaxNorm_x(x_winTrain)
    #trainMin_y, trainMax_y, y_winTrain = minMaxNorm_y(y_winTrain,yMinValue,yRefValue)

    testMin_x, testMax_x, x_winTest   = minMaxNorm_x(x_winTest)
    #testMin_y, testMax_y, y_winTest   = minMaxNorm_y(y_winTest  ,yMinValue,yRefValue)

    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm = [],[],[],[]

    for i in range(len(trainMax_x)):
      if trainMax_x[i] == 0.:
        trainMax_x[i] = 1.
    for i in range(len(testMax_x)):
      if testMax_x[i] == 0.:
        testMax_x[i] = 1.
    
    #yRefValue = np.amax(y_winTrain)
    #trainMax_y.fill(yRefValue)
    #testMax_y.fill(yRefValue)
    for i in range(len(x_winTrain)):
      x_winTrain_norm.append( normalise_data_refValue(trainMax_x[i], x_winTrain[i]) )
      if config.ydiffs!='on':
        y_winTrain_norm.append( normalise_data_refValue(trainMax_x[i], y_winTrain[i]) )
      else:
        y_winTrain_norm = y_winTrain.copy()

    for j in range(len(x_winTest)):
      x_winTest_norm.append(  normalise_data_refValue(testMax_x[j] , x_winTest[j])  )
      if config.ydiffs!='on':
        y_winTest_norm.append(  normalise_data_refValue(testMax_x[j] , y_winTest[j])  )
      else:
        y_winTest_norm = y_winTest.copy()
    
    print y_winTrain_norm.shape
    for i in range(len(y_winTrain_norm)):
      if y_winTrain_norm[i]>35. or y_winTrain_norm[i]<-35.:
        y_winTrain_norm[i]=0.
    
    for i in range(len(y_winTest_norm)):
      if y_winTest_norm[i]>35. or y_winTest_norm[i]<-35.:
        y_winTest_norm[i]=0.

        
    print 'np.amax()',np.amax(y_winTrain_norm)

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
    #return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainMax_x),np.array(trainMin_x),np.array(trainMax_y),np.array(trainMin_y),np.array(testMax_x),np.array(testMin_x),np.array(testMax_y),np.array(testMin_y)
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainMax_x),np.array(trainMin_x),np.array(testMax_x),np.array(testMin_x)
  
#=========================================================================================

  
