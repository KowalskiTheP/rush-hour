import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys

## loading data from a CSV file in a pandas DataFrame. 
## The diff between the values in dateColumn and the refdate gets also calculated and added.
## If no date is aviable, dataColumn has to be 'None'
def load_fromCSV(csvFile, decimal , seperator, header, dateColumn):
  df=pd.read_csv(csvFile, decimal=decimal ,sep=seperator, header=header)

  if dateColumn != 'None':
    ## Some definitions and initialisation
    refdate = '01.01.1960'
    date_format = "%d.%m.%Y"
    date_list = []
    b = datetime.strptime(refdate, date_format)
    ## Getting the diff between ref and data date
    for i in range(len(df)):
      a = datetime.strptime(df.loc[i,dateColumn], date_format)
      date_list.append(str(a-b))
      date_list[-1] = [int(s) for s in date_list[-1].split() if s.isdigit() ]
    date_list = np.array(date_list).flatten()
    df['days'] = date_list   
  return df


## Creating a numpy array with all data from the selected columns 
## of the previosly initialised dataFrame
def getDataSet(dataframe, columns, trainTestSplit):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  
  ############### maybe dont split train and test set randomly??? sounds crazy with a time series?
  #trainSet, testSet = train_test_split(dataSet, test_size = trainTestSplit)
  
  # first try last X percent of the dataset are the test set. PLEASE REVIEW!!!!
  # In principle it works now...but the data partioning has to be refined...COME ON RANDOM DATA GENERATOR!
  lastXpercent = int(np.floor(len(dataframe)*trainTestSplit))
  firstXpercent = len(dataframe) - lastXpercent
  
  trainSet = dataSet[0:firstXpercent]
  testSet = dataSet[firstXpercent:firstXpercent+lastXpercent]

  return trainSet, testSet


def split_data(dataSet, trainTestSplit):
  lastXpercent = int(np.floor(len(dataSet)*trainTestSplit))
  firstXpercent = len(dataSet) - lastXpercent
  trainSet = dataSet[0:firstXpercent]
  testSet = dataSet[firstXpercent:firstXpercent+lastXpercent]
  return trainSet, testSet


## Shifting the data by look_back to create usefull x and y arrays
def shiftData(data, y_column, look_back):
  x = np.zeros((len(data)-look_back,data.shape[1]))
  y = np.zeros((len(data)-look_back,1))
  for i in range(len(data)-look_back):
    x[i] = data[i]
    y[i] = data[i+1,y_column]
  return x, y

## The single windows will be the samples for the model
def get_windows(x,y,winLength):
  x_train, y_train = [], []
  for i in range(len(x)-(winLength+1)):
    x_train.append(x[i:i+winLength])
    y_train.append(y[i+winLength-1])
  return np.array(x_train), np.array(y_train)

###############################################

def normalise_data(x_data, y_data):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaler.fit(x_data)
  x_scaled = scaler.transform(x_data)
  y_scaled = scaler.transform(y_data)
  return x_scaled, y_scaled, scaler

###############################################

def make_windowed_data(dataframe, config):
  ''' makes the windowed dataset from a appropriate dataframe'''
  
  dataSetTrain, dataSetTest = getDataSet(dataframe, config['columns'], float(config['traintestsplit']))
  dataSetTrain, dataSetTest, scaler = normalise_data(dataSetTrain, dataSetTest)
  x_fullTrain, y_fullTrain = shiftData(dataSetTrain, config['y_column'], int(config['look_back']))
  x_winTrain, y_winTrain = get_windows(x_fullTrain,y_fullTrain,int(config['winlength']))
  x_fullTest, y_fullTest = shiftData(dataSetTest, config['y_column'], int(config['look_back']))
  x_winTest, y_winTest = get_windows(x_fullTest,y_fullTest,int(config['winlength']))
  
  return x_winTrain, y_winTrain, x_winTest, y_winTest, scaler

###############################################

def getDataSet_noSplit(dataframe, columns):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  return dataSet


# loading data from csv and sequenzilce it ([[day_x1,startStock],[day_x1,endStock],[day_x2,startStock],[day_x2,endStock],...])

def getDataSet_noSplit_seq(dataframe, columns):
  dataSet = np.zeros((len(dataframe)*2, 2))
  j=0
  for i in xrange(0,len(dataframe)*2,2):
    dataSet[i,0] = dataframe.iloc[j,-1]
    dataSet[i,1] = dataframe.iloc[j,columns[0]]
    dataSet[i+1,0] = dataframe.iloc[j,-1]
    dataSet[i+1,1] = dataframe.iloc[j,columns[1]]
    j=j+1
  print 'Data np.array:\n', dataSet, '\n'
  return dataSet

###############################################

def normalise_data_refValue(refValue,data):
  normData = []
  for i in range(len(data)):
    normData.append( data[i]/refValue )
  return np.array(normData)

###############################################
  
def denormalise_data_refValue(refValue,normData):
  denormData = []
  for i in range(len(normData)):
    denormData.append( refValue * normData[i] )
  return np.array(denormData)

###############################################

def get_windows_andShift(x,winLength,look_back,outDim):
  x_train, y_train = [], []
  for i in range(len(x)-(winLength+outDim)):
    x_train.append(x[i:i+winLength])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),-1])
  print np.array(y_train)
  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

###############################################

def get_windows_andShift_seq(x,winLength,look_back,outDim,y_column):
  x_train, y_train = [], []
  #Porbably important bugfix
  #Old: for i in xrange(0,len(x)-(winLength+outDim),2): 
  for i in xrange(0,len(x)-(winLength+outDim),2):
    x_train.append(x[i:i+winLength])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),y_column])
  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

###############################################

def get_windows_andShift_seq_hourly(x,winLength,look_back,outDim,y_column):
  x_train, y_train = [], []

  #Porbably important bugfix
  #Old: for i in xrange(0,len(x)-(winLength+outDim),2): 
  for i in xrange(0,len(x)-(winLength+outDim),1):
    x_train.append(x[i:i+winLength])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),y_column])

  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

###############################################

def get_windows_andShift_seq_hourly_timeDist(x,winLength,look_back,outDim,y_column):
  x_train, y_train = [], []

  #Porbably important bugfix
  #Old: for i in xrange(0,len(x)-(winLength+outDim),2): 
  for i in xrange(0,len(x)-(2*winLength+outDim),1):
    x_train.append(x[i:i+winLength])
    #y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),y_column])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+winLength-1+look_back+outDim-1),y_column])

  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),winLength,outDim))

##############################################

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

###############################################

def make_windowed_data_normOnFull(dataframe, config):
  refValue = float(config['refvalue'])
  winL = int(config['winlength'])
  lookB = int(config['look_back'])
  dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
#  dataSet_Full_norm =normalise_data_refValue(dataSet_Full[0,-1],dataSet_Full)
  dataSet_Full_norm = normalise_data_refValue(refValue,dataSet_Full)
  dataSetTrain_norm, dataSetTest_norm = split_data(dataSet_Full_norm, float(config['traintestsplit']))
  x_winTrain_norm, y_winTrain_norm = get_windows_andShift(dataSetTrain_norm, winL, lookB)
  x_winTest_norm, y_winTest_norm = get_windows_andShift(dataSetTest_norm, winL, lookB)
  return  x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm
 
##############################################

def make_windowed_data_withSplit(dataframe, config):

  refValue = float(config['refvalue'])
  winL = int(config['winlength'])
  lookB = int(config['look_back'])
  xDim = len(config['columns'])
  yDim = int(config['outputdim'])
  y_column = int(config['y_column'])
  dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
  
  dataSetTrain, dataSetTest = split_data(dataSet_Full, float(config['traintestsplit']))

  x_winTrain, y_winTrain = get_windows_andShift_seq_hourly(dataSetTrain, winL, lookB,yDim,y_column)
  
  x_winTest, y_winTest = get_windows_andShift_seq_hourly(dataSetTest, winL, lookB,yDim,y_column)
  

 
  # Deleting evening to morning predictions
  x_tmp, y_tmp = [],[]
  for i in range(len(x_winTrain)):
    if x_winTrain[i,-1,1] != 18:
      x_tmp.append(x_winTrain[i])
      y_tmp.append(y_winTrain[i])
  x_winTrain = np.array(x_tmp)
  y_winTrain = np.array(y_tmp)
  
  #x_tmp, y_tmp = [],[]
  #for i in range(len(x_winTest)):
    #if x_winTest[i,-1,1] != 18:
      #x_tmp.append(x_winTest[i])
      #y_tmp.append(y_winTest[i])
  #x_winTest = np.array(x_tmp)
  #y_winTest = np.array(y_tmp)
  
  #print 'x_winTest[0]:\n', x_winTest[1]
  
  if config['normalise'] == '3':
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm,trainRef,testRef = [],[],[],[],[],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1],x_winTrain[i]))
      y_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1,y_column],y_winTrain[i]))
      trainRef.append(x_winTrain[i,-1])
    for i in range(len(y_winTest)):
      x_winTest_norm.append( normalise_data_refValue(x_winTest[i,-1],x_winTest[i]))
      y_winTest_norm.append( normalise_data_refValue(x_winTest[i,-1,y_column],y_winTest[i]))
      testRef.append(x_winTest[i,-1])
    
  if config['normalise'] == '4':
    trainMin,trainMax,x_winTrain,y_winTrain = minMaxNorm(x_winTrain,y_winTrain,y_column,yNorm=True)
    testMin ,testMax ,x_winTest ,y_winTest  = minMaxNorm(x_winTest ,y_winTest ,y_column,yNorm=True)
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm = [],[],[],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append( normalise_data_refValue(trainMax[i]         ,x_winTrain[i]) )
      y_winTrain_norm.append( normalise_data_refValue(trainMax[i,y_column],y_winTrain[i]) )
    for j in range(len(y_winTest)):
      x_winTest_norm.append(  normalise_data_refValue(testMax[j]          ,x_winTest[j])  )
      y_winTest_norm.append(  normalise_data_refValue(testMax[j,y_column] ,y_winTest[j])  )
      
  
  x_winTrain_norm = np.reshape(np.array(x_winTrain_norm),(len(x_winTrain_norm),winL,xDim ))
  y_winTrain_norm = np.reshape(np.array(y_winTrain_norm),(len(y_winTrain_norm),yDim ))
  x_winTest_norm =  np.reshape(np.array(x_winTest_norm) ,(len(x_winTest_norm) ,winL,xDim ))
  y_winTest_norm =  np.reshape(np.array(y_winTest_norm) ,(len(y_winTest_norm) ,yDim ))

  if config['normalise'] == '3':
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainRef), np.array(testRef)
  
  if config['normalise'] == '4':
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainMax),np.array(trainMin),np.array(testMax),np.array(testMin)  
  
##############################################

def make_windowed_data_withSplit_timeDist(dataframe, config):

  refValue = float(config['refvalue'])
  winL = int(config['winlength'])
  lookB = int(config['look_back'])
  xDim = len(config['columns'])
  yDim = int(config['outputdim'])
  y_column = int(config['y_column'])
  dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
  
  dataSetTrain, dataSetTest = split_data(dataSet_Full, float(config['traintestsplit']))

  x_winTrain, y_winTrain = get_windows_andShift_seq_hourly_timeDist(dataSetTrain, winL, lookB,yDim,y_column)
  
  x_winTest, y_winTest = get_windows_andShift_seq_hourly_timeDist(dataSetTest, winL, lookB,yDim,y_column)
  
  print np.shape(y_winTest)
  print np.shape(x_winTest)

 
  # Deleting evening to morning predictions
  x_tmp, y_tmp = [],[]
  for i in range(len(x_winTrain)):
    if x_winTrain[i,-1,1] != 18:
      x_tmp.append(x_winTrain[i])
      y_tmp.append(y_winTrain[i])
  x_winTrain = np.array(x_tmp)
  y_winTrain = np.array(y_tmp)

  #x_tmp, y_tmp = [],[]
  #for i in range(len(x_winTest)):
    #if x_winTest[i,-1,1] != 18:
      #x_tmp.append(x_winTest[i])
      #y_tmp.append(y_winTest[i])
  #x_winTest = np.array(x_tmp)
  #y_winTest = np.array(y_tmp)
  
  #print 'x_winTest[0]:\n', x_winTest[1]
  
  if config['normalise'] == '3':
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm,trainRef,testRef = [],[],[],[],[],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1],x_winTrain[i]))
      y_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1,y_column],y_winTrain[i]))
      trainRef.append(x_winTrain[i,-1])
    for i in range(len(y_winTest)):
      x_winTest_norm.append( normalise_data_refValue(x_winTest[i,-1],x_winTest[i]))
      y_winTest_norm.append( normalise_data_refValue(x_winTest[i,-1,y_column],y_winTest[i]))
      testRef.append(x_winTest[i,-1])
    
  if config['normalise'] == '4':
    trainMin,trainMax,x_winTrain,y_winTrain = minMaxNorm(x_winTrain,y_winTrain,y_column,yNorm=True)
    testMin ,testMax ,x_winTest ,y_winTest  = minMaxNorm(x_winTest ,y_winTest ,y_column,yNorm=True)
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm = [],[],[],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append( normalise_data_refValue(trainMax[i]         ,x_winTrain[i]) )
      y_winTrain_norm.append( normalise_data_refValue(trainMax[i,y_column],y_winTrain[i]) )
    for j in range(len(y_winTest)):
      x_winTest_norm.append(  normalise_data_refValue(testMax[j]          ,x_winTest[j])  )
      y_winTest_norm.append(  normalise_data_refValue(testMax[j,y_column] ,y_winTest[j])  )
      
  
  x_winTrain_norm = np.reshape(np.array(x_winTrain_norm),(len(x_winTrain_norm),winL,xDim ))
  y_winTrain_norm = np.reshape(np.array(y_winTrain_norm),(len(y_winTrain_norm),yDim ))
  x_winTest_norm =  np.reshape(np.array(x_winTest_norm) ,(len(x_winTest_norm) ,winL,xDim ))
  y_winTest_norm =  np.reshape(np.array(y_winTest_norm) ,(len(y_winTest_norm) ,yDim ))

  if config['normalise'] == '3':
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainRef), np.array(testRef)
  
  if config['normalise'] == '4':
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainMax),np.array(trainMin),np.array(testMax),np.array(testMin)  
  
##############################################
  
def make_windowed_data_noSplit(dataframe, config):
  
  refValue = float(config['refvalue'])
  winL = int(config['winlength'])
  lookB = int(config['look_back'])
  xDim = len(config['columns'])
  yDim = int(config['outputdim'])
  y_column = int(config['y_column'])
  dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
  
  x_winTrain, y_winTrain = get_windows_andShift_seq_hourly(dataSet_Full, winL, lookB,yDim,y_column)

  print 'x_winTrain[101]', x_winTrain[101]

  x_tmp, y_tmp = [],[]
  for i in range(len(x_winTrain)):
    if x_winTrain[i,-1,1] != 18:
      x_tmp.append(x_winTrain[i])
      y_tmp.append(y_winTrain[i])
  x_winTrain = np.array(x_tmp)
  y_winTrain = np.array(y_tmp)
  
  if config['normalise'] == '3':
    x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm,trainRef,testRef = [],[],[],[],[],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1],x_winTrain[i]))
      y_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1,y_column],y_winTrain[i]))
      trainRef.append(x_winTrain[i,-1])       
    for i in range(len(x_winTrain)-100,len(x_winTrain),1):
      x_winTest_norm.append(x_winTrain_norm.pop(-1))
      y_winTest_norm.append(y_winTrain_norm.pop(-1))
      testRef.append(trainRef.pop(-1))

  if config['normalise'] == '4':
    trainMin,trainMax,x_winTrain,y_winTrain = minMaxNorm(x_winTrain,y_winTrain,y_column,yNorm=True)
    x_winTrain_norm, y_winTrain_norm = [],[]
    for i in range(len(y_winTrain)):
      x_winTrain_norm.append( normalise_data_refValue(trainMax[i]         ,x_winTrain[i]) )
      y_winTrain_norm.append( normalise_data_refValue(trainMax[i,y_column],y_winTrain[i]) )
    trainMin = trainMin.tolist()
    trainMax = trainMax.tolist()
    x_winTest_norm, y_winTest_norm,testMax,testMin = [],[],[],[]
    for i in range(len(x_winTrain)-100,len(x_winTrain),1):
      x_winTest_norm.append(x_winTrain_norm.pop(-1))
      y_winTest_norm.append(y_winTrain_norm.pop(-1))
      testMin.append(trainMin.pop(-1))
      testMax.append(trainMax.pop(-1))
  
  x_winTrain_norm = np.reshape(np.array(x_winTrain_norm),(len(x_winTrain_norm),winL,xDim ))
  y_winTrain_norm = np.reshape(np.array(y_winTrain_norm),(len(y_winTrain_norm),yDim ))
  x_winTest_norm =  np.reshape(np.array(x_winTest_norm) ,(len(x_winTest_norm) ,winL,xDim ))
  y_winTest_norm =  np.reshape(np.array(y_winTest_norm) ,(len(y_winTest_norm) ,yDim ))
  
  #print x_winTrain_norm[0]
  #print x_winTest_norm[0]
  
  if config['normalise'] == '3':
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainRef), np.array(testRef)
  
  if config['normalise'] == '4':
    return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainMax),np.array(trainMin),np.array(testMax),np.array(testMin)  
