import time
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras keras.layers.merge import Multiply
from keras.layers.core import *
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import TimeDistributed
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from tabulate import tabulate
import copy

def safe_model(model, conf):
  # serialize model to JSON
  model_json = model.to_json()
  with open(conf.jsonfile, "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(conf.modelfile)
  print("Saved model to disk")

def load_model(conf):
  # load json and create model
  json_file=open(conf.jsonfile, 'r')
  loaded_model_json=json_file.read()
  json_file.close()
  loaded_model=model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(conf.modelfile)
  loaded_model.compile(loss=conf.loss, optimizer=conf.optimiser)
  print("Loaded and compiled model from disk")
  return loaded_model

def stock_loss(y_true, y_pred):
  alpha=10.
  loss=K.switch(K.less(y_true * y_pred, 0),
                  alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true),
                  K.abs(y_true - y_pred)
        )
  return K.mean(loss, axis=-1)

def build_model(conf):
  '''builds model that is specified in conf'''
  start=time.time()
  if conf.verbosity < 2:
    print "building model"


  inputs=Input(shape=(conf.winlength,conf.inputdim,))

  if conf.cnn == 'on':
    print 'cnn on'
    cnn=Conv1D(filters=32, kernel_size=7, padding='causal',input_shape=(None,conf.inputdim), activation='relu', name='cnn1_1')(inputs)
    cnn=Conv1D(filters=32, kernel_size=7, padding='causal', activation='relu', name='cnn1_2')(cnn)
    cnn=MaxPooling1D(pool_size=2, name='max_pool1')(cnn)
    if conf.batchnorm == 'on':
      cnn=BatchNormalization()(cnn)
    cnn=Conv1D(filters=16, kernel_size=5, padding='causal', activation='relu', name='cnn2_1')(cnn)
    cnn=Conv1D(filters=16, kernel_size=5, padding='causal', activation='relu', name='cnn2_2')(cnn)
    cnn=MaxPooling1D(pool_size=2, name='max_pool2')(cnn)
    if conf.batchnorm == 'on':
      cnn=BatchNormalization()(cnn)
    cnn=Conv1D(filters=8, kernel_size=3, padding='causal', activation='relu', name='cnn3_1')(cnn)
    cnn=Conv1D(filters=8, kernel_size=3, padding='causal', activation='relu', name='cnn3_2')(cnn)
    cnn=MaxPooling1D(pool_size=2, name='max_pool3')(cnn)
    if conf.batchnorm == 'on':
      cnn=BatchNormalization()(cnn)

  if conf.timedistributed == 'on':
    if conf.verbosity > 2:
      print 'timedistributed on'
    returnSequences=True
  else:
    if conf.verbosity > 2:
      print 'timedistributed off'
    returnSequences=False

  # first layer is special, gets build by hand
  if isinstance(conf['neuronsperlayer'], list) == True and len(conf.neuronsperlayer) > 1:
    if conf.verbosity > 2:
      print 'more than one layer'
      print 'layer 0: ',conf.neuronsperlayer[0]
    if conf.cnn == 'on':
      if conf.verbosity > 2:
        print 'cnn on'
      if conf.bidirect == 'on':
        if conf.verbosity > 2:
          print 'bidirect on'
        lstm_encode=Bidirectional(LSTM(conf.neuronsperlayer[0],
                                   activation=conf.activationperlayer[0],
                                   return_sequences=True,
                                   recurrent_activation=conf.recurrentactivation[0],
                                   dropout=conf.dropout[0],
                                   recurrent_dropout=conf.dropout[0]))(cnn)
      else:
        if conf.verbosity > 2:
          print 'bidirect off'
        lstm_encode=LSTM(conf.neuronsperlayer[0],
                         activation=conf.activationperlayer[0],
                         return_sequences=True,
                         recurrent_activation=conf.recurrentactivation[0],
                         dropout=conf.dropout[0],
                         recurrent_dropout=conf.dropout[0])(cnn)
    else:
      if conf.verbosity > 2:
        print 'cnn off'
      if conf.bidirect == 'on':
        if conf.verbosity > 2:
          print 'bidirect on'
        lstm_encode=Bidirectional(LSTM(conf.neuronsperlayer[0],
                                       activation=conf.activationperlayer[0],
                                       return_sequences=True,
                                       recurrent_activation=conf.recurrentactivation[0],
                                       dropout=conf.dropout[0],
                                       recurrent_dropout=conf.dropout[0]))(inputs)
      else:
        if conf.verbosity > 2:
          print 'bidirect off'
        lstm_encode=LSTM(conf.neuronsperlayer[0],
                         activation=conf.activationperlayer[0],
                         return_sequences=True,
                         recurrent_activation=conf.recurrentactivation[0],
                         dropout=conf.dropout[0],
                         recurrent_dropout=conf.dropout[0])(inputs)


    if len(conf.neuronsperlayer) > 2:
      if conf.verbosity > 2:
        print 'build more than 2 layer'
        print 'if attention is on then it will be applied before \n the last lstm layer'
      for i in xrange(1,len(conf.neuronsperlayer)-1):
        if conf.bidirect == 'on':
          if conf.verbosity > 2:
            print 'bidirect on'
          lstm_encode=Bidirectional(LSTM(conf.neuronsperlayer[i],
                                         activation=conf.activationperlayer[i],
                                         return_sequences=returnSequences,
                                         recurrent_activation=conf.recurrentactivation[i],
                                         dropout=conf.dropout[i],
                                         recurrent_dropout=conf.dropout[i]))(lstm_encode)
        else:
          if conf.verbosity > 2:
            print 'bidirect off'
          lstm_encode=LSTM(conf.neuronsperlayer[i],
                           activation=conf.activationperlayer[i],
                           return_sequences=returnSequences,
                           recurrent_activation=conf.recurrentactivation[i],
                           dropout=conf.dropout[i],
                           recurrent_dropout=conf.dropout[i])(lstm_encode)
    if conf.attention == 'on':
      if conf.verbosity > 2:
        print 'attention on'
      num_hidden=int(lstm_encode.shape[2])
      attention=Permute((2,1))(lstm_encode)
      # not sure if the reshape is necessary? (FM)
      attention=Reshape((num_hidden,-1))(attention)
      # maybe a timedistributed dense layer with 1 neuron should also do the trick? (FM)
      attention=Dense(conf.winlength, activation='softmax')(attention)
      attention_probability=Permute((2,1))(attention)
      lstm_encode=Multiply([lstm_encode, attention_probability])
    else:
      if conf.verbosity > 2:
        print 'attention off'

    if conf.bidirect == 'on':
      if conf.verbosity > 2:
        print 'bidirect on'
      lstm_decode=Bidirectional(LSTM(conf.neuronsperlayer[-1],
                                     activation=conf.activationperlayer[-1],
                                     return_sequences=returnSequences,
                                     recurrent_activation=conf.recurrentactivation[-1],
                                     dropout=conf.dropout[-1],
                                     recurrent_dropout=conf.dropout[-1]))(lstm_encode)
    else:
      if conf.verbosity > 2:
        print 'bidirect off'
      lstm_decode=LSTM(conf.neuronsperlayer[-1],
                       activation=conf.activationperlayer[-1],
                       return_sequences=returnSequences,
                       recurrent_activation=conf.recurrentactivation[-1],
                       dropout=conf.dropout[-1],
                       recurrent_dropout=conf.dropout[-1])(lstm_encode)

    if conf.batchnorm == 'on':
      lstm_decode=BatchNormalization()(lstm_decode)

  else:
    if conf.verbosity > 2:
      print 'only one layer'
      print 'no attention attention implemented'
    if conf.cnn == 'on':
      if conf.verbosity > 2:
        print 'cnn is on'
      if conf.bidirect == 'on':
        lstm_decode=Bidirectional(LSTM(conf.neuronsperlayer,
                                       activation=conf.activationperlayer,
                                       return_sequences=returnSequences,
                                       recurrent_activation=conf.recurrentactivation,
                                       dropout=conf.dropout,
                                       recurrent_dropout=conf.dropout))(cnn)
      else:
        lstm_decode=LSTM(conf.neuronsperlayer,
                         activation=conf.activationperlayer,
                         return_sequences=returnSequences,
                         recurrent_activation=conf.recurrentactivation,
                         dropout=conf.dropout,
                         recurrent_dropout=conf.dropout)(cnn)

    else:
      if conf.verbosity > 2:
        print 'cnn is off'
      if conf.bidirect == 'on':
        if conf.verbosity > 2:
          print 'bidirect on'
        lstm_decode=Bidirectional(LSTM(conf.neuronsperlayer,
                                       activation=conf.activationperlayer,
                                       return_sequences=returnSequences,
                                       recurrent_activation=conf.recurrentactivation,
                                       dropout=conf.dropout,
                                       recurrent_dropout=conf.dropout))(inputs)
      else:
        if conf.verbosity > 2:
          print 'bidirect off'
        lstm_decode=LSTM(conf.neuronsperlayer,
                         activation=conf.activationperlayer,
                         return_sequences=returnSequences,
                         recurrent_activation=conf.recurrentactivation,
                         dropout=conf.dropout,
                         recurrent_dropout=conf.dropout)(inputs)

    if conf.batchnorm == 'on':
      lstm_decode=BatchNormalization()(lstm_decode)

  if conf.timedistributed == 'on':
    dense=TimeDistributed(Dense(units=conf.outputlength,
                activation='linear'))(lstm_decode)
  else:
    dense=Dense(units=conf.outputlength,
                activation='linear')(lstm_decode)

  model=Model(inputs=inputs, outputs=dense)

  if conf.verbosity >= 2:
    print '> Build time : ', time.time() - start
    model.summary()

  start=time.time()
  #!!!!!!!!implement learning Rate scheduler!!!!!!!!
  if conf.optimiser == 'adam':
      opt=Adam(lr=conf.learningrate,
               decay=conf.decay)
  #!!!!!!!!implement working custom loss!!!!!!!!
  if conf.loss == 'stock_loss':
    model.compile(loss=stock_loss, optimizer=opt)
  else:
    model.compile(loss=conf.loss, optimizer=opt)

  #if conf.timedistributed == 'on':
    #model=TimeDistributed(model)

  if conf.verbosity > 2:
    print '> Compilation Time : ', time.time() - start
  return model

###############################################

def predict_point_by_point(model, data):
  #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
  predicted=model.predict(data)
  predicted=np.reshape(predicted, (predicted.size,))
  return predicted

###############################################

def plot_data(true_data, pred_data, title='Your data'):
  '''makes simple plots of the evaluated data, nothing fancy'''
  
  #plt.ion()
  
  plt.title(title)
  plt.plot(true_data, ls='--', linewidth=2, color='tomato')
  plt.plot(pred_data, linewidth=2, color='indigo')
  tomato_patch=mpatches.Patch(color='tomato', label='true data')
  indigo_patch=mpatches.Patch(color='indigo', label='pred. data')
  plt.legend(handles=[tomato_patch,indigo_patch])
  axes=plt.gca()
  plt.autoscale(enable=True, axis='y')
  plt.show()

###############################################

def eval_model(test_x, test_y, trainedModel, config, tableHeader):
  '''calculate some core metrics for model evaluation'''
  
  test_y_shape=test_y.shape
  score=trainedModel.evaluate(test_x, test_y, batch_size=int(config.batchsize))
  pred=predict_point_by_point(trainedModel, test_x)
  test_y=test_y.flatten()
  rp, rp_P=stats.pearsonr(pred,test_y)
  rs, rs_P=stats.spearmanr(pred,test_y)
  sd=np.std(pred-test_y)
  print '------', tableHeader, '------'
  print tabulate({"metric": ['test loss', 'Rp', 'Rs', 'SD'],"model": [score, rp, rs, sd]}, headers="keys", tablefmt="orgtbl")
  np.savetxt(config.predictionfile, np.column_stack((pred, test_y)), delimiter=' ')
  
  return pred
    
###############################################

def get_random_hyperparameterset(conf):
  '''draws a random hyperparameter set when called'''
  #np.random.seed(seed=int(time.time()))
  config1=copy.deepcopy(conf)
  params={}
  
  
  if isinstance(config1.nlayer_tune, list) is True:
    params['nlayer_tune']=int(config1.nlayer_tune[np.random.random_integers(0,len(config1.nlayer_tune)-1)])
  else:
    params['nlayer_tune']=int(config1.nlayer_tune)
  
  if isinstance(config1.actlayer_tune, list) is True:
    params['actlayer_tune']=str(config1.actlayer_tune[np.random.random_integers(0,len(config1.actlayer_tune)-1)])
  else:
    params['actlayer_tune']=str(config1.actlayer_tune)
  
  if isinstance(config1.nhiduplayer_tune, list) is True:
    params['nhiduplayer_tune']=int(config1.nhiduplayer_tune[np.random.random_integers(0,len(config1.nhiduplayer_tune)-1)])
  else:
    params['nhiduplayer_tune']=int(config1.nhiduplayer_tune)
  
  if isinstance(config1.dropout_tune, list) is True:
    params['dropout_tune']=float(config1.dropout_tune[np.random.random_integers(0,len(config1.dropout_tune)-1)])
  else:
    params['dropout_tune']=float(config1.dropout_tune)

  if isinstance(config1.recactlayer_tune, list) is True:
    params['recactlayer_tune']=str(config1.recactlayer_tune[np.random.random_integers(0,len(config1.recactlayer_tune)-1)])
  else:
    params['recactlayer_tune']=str(config1.recactlayer_tune)
  
  temp=[]
  temp1=[]
  temp2=[]
  temp3=[]
  for i in xrange(0,params['nlayer_tune']):
    
    temp.append(params['actlayer_tune'])
    temp1.append(params['nhiduplayer_tune'])
    temp2.append(params['dropout_tune'])
    temp3.append(params['recactlayer_tune'])
  
  config1.neuronsperlayer=temp1
  config1.activationperlayer=temp
  config1.dropout=temp2
  config1.recurrentactivation=temp3
  config1.learningrate=float(config1.lr_tune[np.random.random_integers(0,len(config1.lr_tune)-1)])
  config1.batchsize=int(config1.batchsize_tune[np.random.random_integers(0,len(config1.batchsize_tune)-1)])
  config1.batchnorm=str(config1.batchnorm_tune[np.random.random_integers(0,len(config1.batchnorm_tune)-1)])
  
  #print config1.neuronsperlayer
  #print config1.activationperlayer
  #print config1.dropout
  #print config1.learningrate
  #print config1.batchsize
  #print config1.batchnorm
  
  return config1

###############################################

def run_nn(epochs, temp_config, X_train, Y_train):
  '''builds and the runs the specified model, after that it returns the last loss'''
  #print temp_config.neuronsperlayer
  model=build_model(temp_config)
  
  print '*****'
  print temp_config.neuronsperlayer
  print temp_config.activationperlayer
  print temp_config.dropout
  print '*****'
  
  hist=model.fit(X_train, Y_train, epochs=epochs, batch_size=int(temp_config.batchsize), verbose=0)
  
  last_loss=hist.history.loss[-1]
  
  return last_loss

###############################################

def write_params(conf, filename):
  '''writes the dictionary defined in params to a file'''
  
  with open(filename, 'w') as f:
    for key, value in conf.items():
      f.write('%s: %s\n' % (key, value))
      
###############################################

def hypertune(X_train, Y_train, config):
  '''hyperband algorithm adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html'''
  
  history_list=[]
  
  start=time.time()
  print '> hyperparameter tuning through the hyperband algorithm will be done'
  
  max_iter=21  # maximum iterations/epochs per configuration
  eta=3 # defines downsampling rate (default=3)
  logeta=lambda x: np.log(x)/np.log(eta)
  s_max=int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
  B=(s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

  #### Begin Finite Horizon Hyperband outerloop. Repeat indefinetely.
  for s in reversed(range(s_max+1)):

    n=int(np.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
    r=max_iter*eta**(-s) # initial number of iterations to run configurations for

    #### Begin Finite Horizon Successive Halving with (n,r)
    
    T=[ get_random_hyperparameterset(config) for i in range(n) ]
    #print T
    for i in range(s+1):
      # Run each of the n_i configs for r_i iterations and keep best n_i/eta
      n_i=n*eta**(-i)
      r_i=r*eta**(i)
      print '######################################'
      print 'keep best: ', n_i/eta
      print 'number of epochs: ', int(r_i)
      print 'number of configs: ', len(T)
      print '######################################'
      #for t in T:
      #  print t
      val_losses=[ run_nn(int(r_i), t, X_train, Y_train) for t in T ]
      T=[ T[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
    
    print np.argsort(val_losses)[0:int( n_i/eta )]
    history_list.append(T)
    print history_list
  
  
  filename=str(config.bestparams)
  write_params(T[0], filename)
  print np.argsort(val_losses)[0:int( n_i/eta )]
  print history_list
  
  print '> hyperparameter tuning took : ', time.time() - start
    
