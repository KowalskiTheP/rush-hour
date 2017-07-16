import readConf
import model
import loadData
import os
import time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=150)
import sys

config = readConf.readINI("../Data/config.conf")
os.environ.TF_CPP_MIN_LOG_LEVEL = str(config.loglevel)



conf.epochs = 20
  
for i in range(3):
  layerList = [10]
    for j in range(i):
      layerList.append(10)
  conf.neuronsPerLayer = layerList

def   
  global_start_time = time.time()
  loadData_start_time = time.time()
  print '> Loading data... '
  
  dataframe = loadData.load_fromCSV('../Data/unitTest_simple.csv', '.', ',', int(config.header), config.datecolumn)
    
  print '> Windowing data...'
  yDim = int(config.outputdim)
  
  if config.normalise == '3':   
      x_winTrain, y_winTrain, x_winTest, y_winTest,trainRef, testRef = loadData.make_windowed_data_withSplit(dataframe,config)
  if config.normalise == '4':
      x_winTrain, y_winTrain, x_winTest, y_winTest,trainMax,trainMin,testMax,testMin = loadData.make_windowed_data_withSplit(dataframe,config)
        
  if config.timedistributed == 'on':
      y_winTrain = np.reshape(y_winTrain, (len(y_winTrain), yDim, 1))
      y_winTest  = np.reshape(y_winTest, (len(y_winTest), yDim, 1))
  
  print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'
  
  if config.tuning == 'on':
    
    #config = model.get_random_hyperparameterset(config)
    model.hypertune(x_winTrain, y_winTrain, config)
    sys.exit()
  
  else:
  
    model1 = model.build_model(config)
    model1.fit(x_winTrain, y_winTrain, int(config.batchsize), int(config.epochs))
    
    jsonFile = str(config.jsonfile)
    modelFile = str(config.modelfile)
    model.safe_model(model1, jsonFile, modelFile)
    loaded_model = model.load_model(jsonFile, modelFile)
  
    if config.evalmetrics == 'on':
      predTest = model.eval_model(x_winTest, y_winTest, loaded_model, config, 'test data')
      predTrain = model.eval_model(x_winTrain, y_winTrain, loaded_model, config, 'train data')
    else:
      predTest = model.predict_point_by_point(loaded_model, x_winTest)
      predTrain = model.predict_point_by_point(loaded_model, x_winTrain)
      print np.column_stack((pred, y_winTest))
    
    predTrain = np.reshape(predTrain,y_winTrain.shape)
    predTest = np.reshape(predTest,y_winTest.shape)
      
    if config.normalise == '3':
      y_column = int(config.y_column)
      for i in range(len(testRef)):
        predTest[i] = (testRef[i,y_column]*predTest[i])
        y_winTest[i] = (testRef[i,y_column]*y_winTest[i])
        x_winTest[i] = (testRef[i]*x_winTest[i])
      for i in range(len(trainRef)):
        y_winTrain[i] = trainRef[i,y_column]*y_winTrain[i]
        predTrain[i] = trainRef[i,y_column]*predTrain[i]
      
    if config.normalise == '4':
      x_winTest_deN = np.copy(x_winTest)
      y_column = int(config.y_column)
      for i in range(len(testMax)):
        predTest[i] = (testMax[i,y_column]*predTest[i]) + testMin[i,y_column]
        y_winTest[i] = (testMax[i,y_column]*y_winTest[i]) + testMin[i,y_column]
        x_winTest[i] = (testMax[i]*x_winTest[i]) + testMin[i]
      for i in range(len(trainMax)):
        y_winTrain[i] = trainMax[i,y_column]*y_winTrain[i] + trainMin[i,y_column]
        predTrain[i] = trainMax[i,y_column]*predTrain[i] + trainMin[i,y_column]
        
  winL = int(config.winlength)
  
  predTest = np.reshape(predTest,(len(y_winTest),yDim,1))
  predTrain = np.reshape(predTrain,(len(y_winTrain),yDim,1))
  y_winTest = np.reshape(y_winTest,(len(y_winTest),yDim,1))
  y_winTrain = np.reshape(y_winTrain,(len(y_winTrain),yDim,1))
  
  if config.timedistributed == 'on':
    for i in range(int(config.look_back)+int(config.winlength)-1,0,-1):
      diffTrain = np.sqrt((predTest[:,-i] - y_winTest[:,-i])**2)
      slopePred = (predTest[:,-i]-predTest[:,-i-1]) 
      slopeTest = (y_winTest[:,-i]-y_winTest[:,-i-1])
      rightSign = 0
      for k in range(len(slopePred)):
        if np.sign(slopePred[k]) == np.sign(slopeTest[k]):
          rightSign = rightSign + 1
      
      print 'Correct Trends [%]: ',  rightSign / float(len(slopePred))
      print 'Mean of pred.-true-diff ('+str(-i)+') :               ', np.mean(diffTrain)
      print 'Standard deviation of pred.-true-diff ('+str(-i)+') : ', np.std(diffTrain) ,'\n'
  else:
    diffTrain = np.sqrt((predTest[:,-i] - y_winTest[:,-i])**2)
    print 'Mean of pred.-true-diff:               ', np.mean(diffTrain)
    print 'Standard deviation of pred.-true-diff: ', np.std(diffTrain)
