import readConf
import model
import loadData
import os
import time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=150)
import sys

#import matplotlib.pyplot as plt

config = readConf.readINI("../Data/config.conf")
yDim = int(config['outputdim'])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(config['loglevel'])
  
global_start_time = time.time()
  
loadData_start_time = time.time()
print '> Loading data... '
  
#dataframe = loadData.load_fromCSV(config['csvfile'], ',', ';', int(config['header']), config['datecolumn'])
dataframe = loadData.load_fromCSV(config['csvfile'], '.', ',', int(config['header']), config['datecolumn'])

if config['windoweddata'] == 'on':
  
  print '> Windowing data..be patient little Padawan'
  
  if config['normalise'] == '1':
    x_winTrain, y_winTrain, x_winTest, y_winTest, scaler = loadData.make_windowed_data(dataframe, config)  
  
  if config['normalise'] == '2':
    refValue = float(config['refvalue'])
    x_winTrain, y_winTrain, x_winTest, y_winTest = loadData.make_windowed_data_normOnFull(dataframe, config) 
    
  if config['split'] == 'on':
    if config['normalise'] == '3':   
      x_winTrain, y_winTrain, x_winTest, y_winTest,trainRef, testRef = loadData.make_windowed_data_withSplit(dataframe,config)
    if config['normalise'] == '4':
      x_winTrain, y_winTrain, x_winTest, y_winTest,trainMax,trainMin,testMax,testMin = loadData.make_windowed_data_withSplit(dataframe,config)
      
  #y_winTrain = np.reshape(y_winTrain, (len(y_winTrain), 29, 1))
  #print x_winTrain.shape
  #print y_winTrain.shape
  
  print 'x_winTrain[1000]',x_winTrain[1]
  print 'y_winTrain[1000]',y_winTrain[1]
  

####### TESTING OF NEW FUNCTIONS######
  #if config['split'] == 'on':
    #if config['normalise'] == '3':   
      #x_winTrain, y_winTrain, x_winTest, y_winTest,trainRef, testRef = loadData.make_windowed_data_withSplit_timeDist(dataframe,config)
    #if config['normalise'] == '4':
      #x_winTrain, y_winTrain, x_winTest, y_winTest,trainMax,trainMin,testMax,testMin = loadData.make_windowed_data_withSplit_timeDist(dataframe,config)

  #sys.exit()
  if config['split'] == 'off':
    if config['normalise'] == '3':   
      x_winTrain, y_winTrain, x_winTest, y_winTest,trainRef, testRef = loadData.make_windowed_data_noSplit(dataframe,config)
    if config['normalise'] == '4':
      x_winTrain, y_winTrain, x_winTest, y_winTest,trainMax,trainMin,testMax,testMin = loadData.make_windowed_data_noSplit(dataframe,config) 
      
  if config['timedistributed'] == 'on':
    y_winTrain = np.reshape(y_winTrain, (len(y_winTrain), yDim, 1))
    y_winTest  = np.reshape(y_winTest, (len(y_winTest), yDim, 1))

else:
  print 'not implemented so far, exiting!'
  sys.exit()
  
print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'

if config['tuning'] == 'on':
  
  #config = model.get_random_hyperparameterset(config)
  model.hypertune(x_winTrain, y_winTrain, config)
  sys.exit()

else:
  
  # build the specified model
  model1 = model.build_model(config)
  
  # train the model
  model1.fit(x_winTrain, y_winTrain, int(config['batchsize']), int(config['epochs']))
  
  jsonFile = str(config['jsonfile'])
  modelFile = str(config['modelfile'])
  model.safe_model(model1, jsonFile, modelFile)
  loaded_model = model.load_model(jsonFile, modelFile)

  # simple predictions or eval metrics
  #y_winTest = y_winTest.flatten()
  #y_winTrain = y_winTrain.flatten()
  
  print 'y_winTrain.shape', y_winTrain.shape  
  print 'y_winTest.shape', y_winTest.shape

  if config['evalmetrics'] == 'on':
    predTest = model.eval_model(x_winTest, y_winTest, loaded_model, config, 'test data')
    predTrain = model.eval_model(x_winTrain, y_winTrain, loaded_model, config, 'train data')
  else:
    predTest = model.predict_point_by_point(loaded_model, x_winTest)
    predTrain = model.predict_point_by_point(loaded_model, x_winTrain)
    print np.column_stack((pred, y_winTest))
  
  predTrain = np.reshape(predTrain,y_winTrain.shape)
  predTest = np.reshape(predTest,y_winTest.shape)
  print 'predTrain.shape', predTrain.shape  
  print 'predTest.shape', predTest.shape
    
  if config['normalise'] == '1':
    predTest = scaler.inverse_transform(predTest)
    y_winTest = scaler.inverse_transform(y_winTest)
    y_winTrain = scaler.inverse_transform(y_winTrain)
    predTrain = scaler.inverse_transform(predTrain)
  
  if config['normalise'] == '2':
    refValue = float(config['refvalue'])
    predTest = loadData.denormalise_data_refValue(refValue,predTest)
    y_winTest = loadData.denormalise_data_refValue(refValue,y_winTest)
    y_winTrain = loadData.denormalise_data_refValue(refValue,y_winTrain)
    predTrain = loadData.denormalise_data_refValue(refValue,predTrain)
  

  
  if config['normalise'] == '3':
    y_column = int(config['y_column'])
    for i in range(len(testRef)):
      predTest[i] = (testRef[i,y_column]*predTest[i])
      y_winTest[i] = (testRef[i,y_column]*y_winTest[i])
      x_winTest[i] = (testRef[i]*x_winTest[i])
    for i in range(len(trainRef)):
      y_winTrain[i] = trainRef[i,y_column]*y_winTrain[i]
      predTrain[i] = trainRef[i,y_column]*predTrain[i]
    
  if config['normalise'] == '4':
    x_winTest_deN = np.copy(x_winTest)
    y_column = int(config['y_column'])
    for i in range(len(testMax)):
      predTest[i] = (testMax[i,y_column]*predTest[i]) + testMin[i,y_column]
      y_winTest[i] = (testMax[i,y_column]*y_winTest[i]) + testMin[i,y_column]
      x_winTest[i] = (testMax[i]*x_winTest[i]) + testMin[i]
    for i in range(len(trainMax)):
      y_winTrain[i] = trainMax[i,y_column]*y_winTrain[i] + trainMin[i,y_column]
      predTrain[i] = trainMax[i,y_column]*predTrain[i] + trainMin[i,y_column]
      

winL = int(config['winlength'])
#==========================
# with some luck, not needed anymore 
#==========================
#saftyCopy = np.copy(y_winTest)
#toPredWin = np.reshape(np.copy(x_winTest_deN[0]),(1,winL,3))
#toPredWin_deN = np.copy(x_winTest_deN[0])
#toPredMin = np.copy(testMin[0])
#toPredMax = np.copy(testMax[0])
#for i in range(19):
  #toPredMin,toPredMax,toPredWin,y_winTest_deN =loadData.minMaxNorm(toPredWin,saftyCopy,y_column,yNorm=True)
  #toPredWin = toPredWin / toPredMax
  #print i
  #print toPredWin
  #tmpPred_deN = np.copy(loaded_model.predict(toPredWin) ) *toPredMax[0,y_column]+toPredMin[0,y_column]
  #tmpPredList.append(np.copy(tmpPred_deN))
  #toPredWin = np.copy(toPredWin)  * toPredMax + toPredMin 
  #for a in range(len(toPredWin[0])-1):
    #toPredWin[0,a,:] = toPredWin[0,a+1,:]
  #toPredWin[0,-1,:] = x_winTest[i+1,-1]*testMax[i+1]+testMin[i+1]
  #toPredWin[0,-1,-1] = tmpPred_deN
  
#tmpPredList = np.array(tmpPredList).flatten()
#print tmpPredList


# If the y-dimension is bigger then one, some additional mambo jambo 
# has to be done to get corresponding y and ^y values

#if yDim > 1:
  #print 'y dimension is bigger then 1'
  #predTrain = np.reshape(predTrain,(len(y_winTrain),yDim))
  #l = list(range(0, yDim-1))
  #lback = list(range((len(y_winTest)-yDim+1),len(y_winTest)))
  #lfull = l + lback
  #y_winTest = np.delete(y_winTest, lfull, 0)
  #y_winTest = np.delete(y_winTest, l, 1)
  #y_winTest = y_winTest.flatten()
  #tmpPredTest = []
  #tmpPredTest_test = []
  #for i in range(len(y_winTest)):
    #tmpY = 0.
    #for j in range(yDim):
      #tmpY = tmpY + predTest[i+j,yDim-1-j]
    #tmpPredTest.append(tmpY/yDim)
  #predTest = np.array(tmpPredTest)

predTest = np.reshape(predTest,(len(y_winTest),yDim,1))
predTrain = np.reshape(predTrain,(len(y_winTrain),yDim,1))
y_winTest = np.reshape(y_winTest,(len(y_winTest),yDim,1))
y_winTrain = np.reshape(y_winTrain,(len(y_winTrain),yDim,1))

tmp_predTest = np.zeros(len(predTest))
tmp_yTest = np.zeros(len(y_winTest))
dif = 0.
for i in range(yDim,len(predTest)):
  for j in range(yDim,0,-1):
    tmp_predTest[i] = tmp_predTest[i] + predTest[i-j,j-1] * 1./j
    dif=dif+1./j
  tmp_predTest[i] = tmp_predTest[i] / dif
  dif = 0.
  tmp_yTest[i] = y_winTest[i,-1]
  
tmp_predTest = np.trim_zeros(tmp_predTest)  
tmp_yTest = np.trim_zeros(tmp_yTest)

print 'x_winTest\n',x_winTest[-5:-1]

print 'predTest\n',predTest[-5:-1]
print 'y_winTest\n',y_winTest[-5:-1]

print 'tmp_predTest\n',tmp_predTest[-5:-1]
print 'tmp_yTest\n',tmp_yTest[-5:-1]

diffTrain = np.sqrt((predTest - y_winTest)**2)
print 'Mean of pred.-true-diff:               ', np.mean(diffTrain)
print 'Standard deviation of pred.-true-diff: ', np.std(diffTrain)

#y_winTrain = y_winTrain.flatten()
#predTrain = predTrain.flatten()

if config['plotting'] == 'on':
  #model.plot_data(y_winTrain, predTrain)
  model.plot_data(tmp_yTest, tmp_predTest)
  model.plot_data(tmp_yTest[-8:-1], tmp_predTest[-8:-1])
  model.plot_data(y_winTest.flatten(), predTest.flatten())
  #model.plot_data(y_winTest[0:len(tmpPredList)], tmpPredList)
  #model.plot_data(x_winTest_deN[-50:-1,-1,2], predTest[-50:-1])

  



