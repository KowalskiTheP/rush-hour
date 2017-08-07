import readConf
import model
import loadData
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from tabulate import tabulate
np.set_printoptions(linewidth=150)
import sys
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

conf = readConf.readINI("../Data/config.conf")
os.environ.TF_CPP_MIN_LOG_LEVEL = str(conf.loglevel)

global_start_time = time.time()
loadData_start_time = time.time()
print '> Loading data... '
#dataframe = loadData.load_fromCSV(conf.csvfile, ',', ';', int(conf.header), conf.datecolumn)
dataframe = loadData.load_fromCSV(conf.csvfile, '.', ',', int(conf.header), conf.datecolumn)


if conf.timeinterval == 1:
  print 'training on minute intervals'

else:
  print 'training on '+str(conf.timeinterval)+' minute intervals'
  cols = list(dataframe.columns.values)
  dfNew = pd.DataFrame()
  for i in range(len(dataframe)):
    if dataframe.iloc[i,0] % conf.timeinterval == 0:
      dfNew = dfNew.append(dataframe.iloc[i,:])

  index = range(0,len(dfNew))
  dataframe = dfNew.reset_index(drop=True)
  dataframe = dataframe[cols]
  print 'dataframe:\n',dataframe[cols]


print '> Windowing data...'
yLen = int(conf.outputlength)

if conf.normalise == 0:
    x_winTrain, y_winTrain, x_winTest, y_winTest = loadData.make_windowed_data_withSplit(dataframe,conf)

if conf.normalise == 3:   
    x_winTrain, y_winTrain, x_winTest, y_winTest, trainRef, testRef = loadData.make_windowed_data_withSplit(dataframe,conf)
if conf.normalise == 4:
    #x_winTrain, y_winTrain, x_winTest, y_winTest,trainMax_x,trainMin_x,trainMax_y,trainMin_y,testMax_x,testMin_x,testMax_y,testMin_y = loadData.make_windowed_data_withSplit(dataframe,conf)
    x_winTrain, y_winTrain, x_winTest, y_winTest,trainMax_x,trainMin_x,testMax_x,testMin_x = loadData.make_windowed_data_withSplit(dataframe,conf)

if conf.timedistributed == 'on':
    y_winTrain = np.reshape(y_winTrain, (len(y_winTrain), yLen, 1))
    y_winTest  = np.reshape(y_winTest, (len(y_winTest), yLen, 1))

for jjj in range(len(x_winTrain)):
  if np.isnan(x_winTrain[jjj]).any() == True:
    print np.isnan(x_winTrain[jjj])

#y_winTestMean=np.mean(y_winTest)
#print 'y_winTestMean', y_winTestMean
#y_winTest=y_winTest/y_winTestMean
#y_winTrain=y_winTrain/y_winTestMean

print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'

if conf.tuning == 'on':

  #conf = model.get_random_hyperparameterset(conf)
  model.hypertune(x_winTrain, y_winTrain, conf)
  sys.exit()

else:
###### BUILDING AND TRAINING OF MODEL

  model1=model.build_model(conf)
  #early stopping stuff
  earlyStopping=EarlyStopping(monitor='val_loss', min_delta=conf.earlystop, patience=30, verbose=2)

  #checkpoint stuff
  saving_interval=np.round(conf.epochs/4).astype(int)
  filen, fileext=os.path.splitext(conf.modelfile)
  filen=filen+'_epoch{epoch:02d}'+fileext
  checkpoint=ModelCheckpoint(filen, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=saving_interval)
  
  #Learning rate schedule
  lr_sched=ReduceLROnPlateau(monitor='val_loss', factor=conf.decay, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)


  if conf.stateful == 'on':
    for epoch in range(conf.epochs):
      mean_training_accuracy = []
      mean_training_loss = []
      for i in range(len(x_winTrain)):
          y = y_winTrain[i]
          x = np.expand_dims(x_winTrain[i], axis=0)
          training_loss,training_acc=model1.train_on_batch(x,y)
          mean_training_accuracy.append(training_acc)
          mean_training_loss.append(training_loss)
      model1.reset_states()
      print('accuracy training = {}'.format(np.mean(mean_training_accuracy)))
      print('loss training = {}'.format(np.mean(mean_training_loss)))
      print('___________________________________')
  else:
    #fitting stuff
    plt.plot(y_winTrain)
    plt.show()

    model1.fit(x_winTrain, y_winTrain, conf.batchsize, conf.epochs, callbacks=[earlyStopping,checkpoint,lr_sched], validation_split=0.1, shuffle=False)


  #save last model
  model.safe_model(model1, conf)
  loaded_model = model.load_model(conf)

######

  # simple predictions or eval metrics
  #y_winTest = y_winTest.flatten()
  #y_winTrain = y_winTrain.flatten()

  if conf.evalmetrics == 'on':
    if conf.stateful == 'on':
      predTest = model.eval_model_stateful(x_winTest, y_winTest, loaded_model, conf, 'test data')
      predTrain = model.eval_model_stateful(x_winTrain, y_winTrain, loaded_model, conf, 'train data')
    else:
      predTest = model.eval_model(x_winTest, y_winTest, loaded_model, conf, 'test data')
      predTrain = model.eval_model(x_winTrain, y_winTrain, loaded_model, conf, 'train data')
  else:
    predTest = model.predict_point_by_point(loaded_model, x_winTest)
    predTrain = model.predict_point_by_point(loaded_model, x_winTrain)
    print np.column_stack((pred, y_winTest))

  predTrain = np.reshape(predTrain,y_winTrain.shape)
  predTest = np.reshape(predTest,y_winTest.shape)

  if conf.normalise == 3:
    for i in range(len(testRef)):
      predTest[i]   = testRef[i]  * predTest[i] 
      y_winTest[i]  = testRef[i]  * y_winTest[i]
    for i in range(len(trainRef)):
      y_winTrain[i] = trainRef[i] * y_winTrain[i]
      predTrain[i]  = trainRef[i] * predTrain[i]
  
  print 'predTrain:\n',predTrain
  if conf.ydiffs!='on':
    if conf.normalise == 4:
      for i in range(len(testMax_x)):
        predTest[i]   = testMax_x[i]  * predTest[i] #  + testMin_x[i]
        y_winTest[i]  = testMax_x[i]  * y_winTest[i]#  + testMin_x[i]
      for i in range(len(trainMax_x)):
        y_winTrain[i] = trainMax_x[i] * y_winTrain[i]# + trainMin_x[i]
        predTrain[i]  = trainMax_x[i] * predTrain[i] # + trainMin_x[i]


winL = int(conf.winlength)

predTest = np.reshape(predTest,(len(y_winTest),yLen,1))
predTrain = np.reshape(predTrain,(len(y_winTrain),yLen,1))
y_winTest = np.reshape(y_winTest,(len(y_winTest),yLen,1))
y_winTrain = np.reshape(y_winTrain,(len(y_winTrain),yLen,1))

if conf.timedistributed == 'on':
  for i in range(conf.look_back+conf.winlength-2,0,-1):
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
  y_winTrain = y_winTrain.flatten()
  predTrain = predTrain.flatten()
  y_winTest = y_winTest.flatten()
  predTest = predTest.flatten()  
  diffTrain = np.sqrt((predTest - y_winTest)**2)
  diffTrain1=predTest-y_winTest
  rp, rp_P=stats.pearsonr(predTest,y_winTest)
  rs, rs_P=stats.spearmanr(predTest,y_winTest)
  mae=np.mean(diffTrain)
  sd_mae=np.std(diffTrain)
  sd=np.std(diffTrain1)
  print '\n------ METRICS ------'
  print tabulate({"metric": ['Rp', 'Rs', 'MAE', 'SD(MAE)', 'SD'],"model": [rp, rs, mae, sd_mae, sd]}, headers="keys", tablefmt="orgtbl")


if conf.plotting == 'on':
  if conf.timedistributed == 'on':
    model.plot_data(y_winTrain[:,-1], predTrain[:,-1])
    model.plot_data(y_winTest[:,-1], predTest[:,-1])
    model.plot_data(y_winTest[::4,-5:-1].flatten(), predTest[::4,-5:-1].flatten())
    model.plot_data(y_winTest[-17,-5:-1], predTest[-17,-5:-1])
    model.plot_data(y_winTest[-13,-5:-1], predTest[-13,-5:-1])
    model.plot_data(y_winTest[-9,-5:-1], predTest[-9,-5:-1])
    model.plot_data(y_winTest[-5,-5:-1], predTest[-5,-5:-1])
    model.plot_data(y_winTest[-1,-5:-1], predTest[-1,-5:-1])
    #model.plot_data(tmp_yTest, tmp_predTest)
    #model.plot_data(tmp_yTest[-8:-1], tmp_predTest[-8:-1])
    #model.plot_data(y_winTest.flatten(), predTest.flatten())
    #model.plot_data(y_winTest[0:len(tmpPredList)], tmpPredList)
    #model.plot_data(x_winTest_deN[-50:-1,-1,2], predTest[-50:-1])
  else:
    model.plot_data(y_winTrain, predTrain)
    model.plot_data(y_winTest, predTest)
    model.plot_data(y_winTest[-50:-1], predTest[-50:-1])





