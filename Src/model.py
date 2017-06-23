import time
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import TimeDistributed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from tabulate import tabulate
import copy

def safe_model(model, jsonFile, modelFile):
  # serialize model to JSON
  model_json = model.to_json()
  with open(jsonFile, "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(modelFile)
  print("Saved model to disk")
  
def load_model(jsonFile, modelFile):
  # load json and create model
  json_file = open(jsonFile, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(modelFile)
  loaded_model.compile(loss='mean_squared_error', optimizer='adam')
  print("Loaded and compiled model from disk")
  return loaded_model


def build_model(params):
  '''builds model that is specified in params'''
  start = time.time()
  if int(params['verbosity']) < 2:
    print "building model"
  
  # build sequential model
  model = Sequential()
  #print len(params['neuronsperlayer'])
  #print params['neuronsperlayer']
  if params['cnn'] == 'on':
    
    model.add(Conv1D(input_shape = (None, int(params['inputdim'])), filters=8, kernel_size=7, padding='causal', activation='relu'))
    #model.add(Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'))
    if str(params['batchnorm']) == 'on':
      model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=4, kernel_size=1, padding='causal', activation='relu'))
    ##model.add(Conv1D(filters=16, kernel_size=1, padding='causal', activation='relu'))
    #if str(params['batchnorm']) == 'on':
      #model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=20, kernel_size=3, padding='causal', activation='relu'))
    #model.add(Conv1D(filters=50, kernel_size=3, padding='causal', activation='relu'))
    #if str(params['batchnorm']) == 'on':
      #model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
  
  # first layer is special, gets build by hand
  if isinstance(params['neuronsperlayer'], list):
      
    if int(params['verbosity']) < 2:
      print 'layer 0: ',params['neuronsperlayer'][0]
    if params['cnn'] == 'on':
      model.add(Bidirectional(LSTM(
        int(params['neuronsperlayer'][0]),
        activation = str(params['activationperlayer'][0]),
        return_sequences=True,
        recurrent_activation = str(params['recurrentactivation'][0]),
        dropout=float(params['dropout'][0]),
        recurrent_dropout=float(params['dropout'][0])
        )
        )
      )
    else:
      model.add(LSTM(
        int(params['neuronsperlayer'][0]),
        input_shape = (None, int(params['inputdim'])),
        activation = str(params['activationperlayer'][0]),
        return_sequences=True,
        recurrent_activation = str(params['recurrentactivation'][0]),
        dropout=float(params['dropout'][0]),
        recurrent_dropout=float(params['dropout'][0])
        )
      )
    if str(params['batchnorm']) == 'on':
      model.add(BatchNormalization())
    
    #model.add(Dropout(float(params['dropout'][0])))
  
    # all interims layer get done by this for loop
    for i in xrange(1,len(params['neuronsperlayer'])-1):
      if int(params['verbosity']) < 2:
        print 'layer ', i, ':', params['neuronsperlayer'][i]
      
      model.add(Bidirectional(LSTM(
        int(params['neuronsperlayer'][i]),
        activation = str(params['activationperlayer'][i]),
        return_sequences=True,
        recurrent_activation = str(params['recurrentactivation'][i]),
        dropout=float(params['dropout'][i]),
        recurrent_dropout=float(params['dropout'][i])
        )
        )
      )
      if str(params['batchnorm']) == 'on':
        model.add(BatchNormalization())
  
      #model.add(Dropout(float(params['dropout'][i])))
    
    #last LSTM layer is special because return_sequences=False
    if str(params['timedistributed']) == 'on':
      returnSequences = True
    else:
      returnSequences = False
    if int(params['verbosity']) < 2:
      print 'last LSTM layer: ',params['neuronsperlayer'][-1]
    model.add(Bidirectional(LSTM(
      int(params['neuronsperlayer'][-1]),
      activation = str(params['activationperlayer'][-1]),
      return_sequences=returnSequences,
      recurrent_activation = str(params['recurrentactivation'][-1]),
      dropout=float(params['dropout'][i]),
      recurrent_dropout=float(params['dropout'][i])
      ),
    merge_mode='ave'
      )
    )
    if str(params['batchnorm']) == 'on':
      model.add(BatchNormalization())
    #model.add(Dropout(float(params['dropout'][-1])))
  
  else:
    print params['neuronsperlayer']
    print int(params['inputdim'])
    print str(params['activationperlayer'])
    print str(params['recurrentactivation'])
    if int(params['verbosity']) < 2:
      print 'layer 0: ',params['neuronsperlayer']
    model.add(LSTM(
      int(params['neuronsperlayer']),
      input_shape = (None, int(params['inputdim'])),
      activation = str(params['activationperlayer']),
      return_sequences=returnSequences,
      recurrent_activation = str(params['recurrentactivation']),
      dropout=float(params['dropout'][i]),
      recurrent_dropout=float(params['dropout'][i])
      )
    )
    if str(params['batchnorm']) == 'on':
      model.add(BatchNormalization())

    #model.add(Dropout(float(params['dropout'][0])))

  #last layer is dense
  if int(params['verbosity']) < 2:
    print 'last layer (dense): ',params['outputdim']
  if str(params['timedistributed']) == 'on':
    model.add(TimeDistributed(Dense(
        #units=int(params['outputdim']),
        #testing parameter
        units=8,
        activation = 'linear'
        )
      )
    )
  else:
    model.add(Dense(
        units=int(params['outputdim']),
        activation = 'linear'
      )
    )


  if int(params['verbosity']) < 2:
    print '> Build time : ', time.time() - start
    model.summary()

  start = time.time()
  if params['optimiser'] == 'adam':
      opt = Adam(lr = float(params['learningrate']),
                 decay=float(params['decay']),
                 )
  model.compile(loss=params['loss'], optimizer=opt)
  
  if int(params['verbosity']) < 2:
    print '> Compilation Time : ', time.time() - start
  return model

###############################################

def predict_point_by_point(model, data):
  #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
  predicted = model.predict(data)
  predicted = np.reshape(predicted, (predicted.size,))
  return predicted

###############################################

def plot_data(true_data, pred_data, title='Your data'):
  '''makes simple plots of the evaluated data, nothing fancy'''
  
  #plt.ion()
  
  plt.title(title)
  plt.plot(true_data, ls='--', linewidth=2, color='tomato')
  plt.plot(pred_data, linewidth=2, color='indigo')
  tomato_patch = mpatches.Patch(color='tomato', label='true data')
  indigo_patch = mpatches.Patch(color='indigo', label='pred. data')
  plt.legend(handles=[tomato_patch,indigo_patch])
  axes = plt.gca()
  plt.autoscale(enable=True, axis='y')
  plt.show()

###############################################

def eval_model(test_x, test_y, trainedModel, config, tableHeader):
  '''calculate some core metrics for model evaluation'''
  
  test_y_shape = test_y.shape
  score = trainedModel.evaluate(test_x, test_y, batch_size=int(config['batchsize']))
  pred = predict_point_by_point(trainedModel, test_x)
  test_y = test_y.flatten()
  rp, rp_P = stats.pearsonr(pred,test_y)
  rs, rs_P = stats.spearmanr(pred,test_y)
  sd = np.std(pred-test_y)
  print '------', tableHeader, '------'
  print tabulate({"metric": ['test loss', 'Rp', 'Rs', 'SD'],"model": [score, rp, rs, sd]}, headers="keys", tablefmt="orgtbl")
  np.savetxt(config['predictionfile'], np.column_stack((pred, test_y)), delimiter=' ')
  
  return pred
    
###############################################

def get_random_hyperparameterset(config):
  '''draws a random hyperparameter set when called'''
  #np.random.seed(seed=int(time.time()))
  config1 = copy.deepcopy(config)
  params = {}
  
  
  if isinstance(config1['nlayer_tune'], list) is True:
    params['nlayer_tune'] = int(config1['nlayer_tune'][np.random.random_integers(0,len(config1['nlayer_tune'])-1)])
  else:
    params['nlayer_tune'] = int(config1['nlayer_tune'])
  
  if isinstance(config1['actlayer_tune'], list) is True:
    params['actlayer_tune'] = str(config1['actlayer_tune'][np.random.random_integers(0,len(config1['actlayer_tune'])-1)])
  else:
    params['actlayer_tune'] = str(config1['actlayer_tune'])
  
  if isinstance(config1['nhiduplayer_tune'], list) is True:
    params['nhiduplayer_tune'] = int(config1['nhiduplayer_tune'][np.random.random_integers(0,len(config1['nhiduplayer_tune'])-1)])
  else:
    params['nhiduplayer_tune'] = int(config1['nhiduplayer_tune'])
  
  if isinstance(config1['dropout_tune'], list) is True:
    params['dropout_tune'] = float(config1['dropout_tune'][np.random.random_integers(0,len(config1['dropout_tune'])-1)])
  else:
    params['dropout_tune'] = float(config1['dropout_tune'])

  if isinstance(config1['recactlayer_tune'], list) is True:
    params['recactlayer_tune'] = str(config1['recactlayer_tune'][np.random.random_integers(0,len(config1['recactlayer_tune'])-1)])
  else:
    params['recactlayer_tune'] = str(config1['recactlayer_tune'])
  
  temp = []
  temp1 = []
  temp2 = []
  temp3 = []
  for i in xrange(0,params['nlayer_tune']):
    
    temp.append(params['actlayer_tune'])
    temp1.append(params['nhiduplayer_tune'])
    temp2.append(params['dropout_tune'])
    temp3.append(params['recactlayer_tune'])
  
  config1['neuronsperlayer'] = temp1
  config1['activationperlayer'] = temp
  config1['dropout'] = temp2
  config1['recurrentactivation'] = temp3
  config1['learningrate'] = float(config1['lr_tune'][np.random.random_integers(0,len(config1['lr_tune'])-1)])
  config1['batchsize'] = int(config1['batchsize_tune'][np.random.random_integers(0,len(config1['batchsize_tune'])-1)])
  config1['batchnorm'] = str(config1['batchnorm_tune'][np.random.random_integers(0,len(config1['batchnorm_tune'])-1)])
  
  #print config1['neuronsperlayer']
  #print config1['activationperlayer']
  #print config1['dropout']
  #print config1['learningrate']
  #print config1['batchsize']
  #print config1['batchnorm']
  
  return config1

###############################################

def run_nn(epochs, temp_config, X_train, Y_train):
  '''builds and the runs the specified model, after that it returns the last loss'''
  #print temp_config['neuronsperlayer']
  model = build_model(temp_config)
  
  print '*****'
  print temp_config['neuronsperlayer']
  print temp_config['activationperlayer']
  print temp_config['dropout']
  print '*****'
  
  hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=int(temp_config['batchsize']), verbose=0)
  
  last_loss = hist.history['loss'][-1]
  
  return last_loss

###############################################

def write_params(params, filename):
  '''writes the dictionary defined in params to a file'''
  
  with open(filename, 'w') as f:
    for key, value in params.items():
      f.write('%s: %s\n' % (key, value))
      
###############################################

def hypertune(X_train, Y_train, config):
  '''hyperband algorithm adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html'''
  
  history_list = []
  
  start = time.time()
  print '> hyperparameter tuning through the hyperband algorithm will be done'
  
  max_iter = 21  # maximum iterations/epochs per configuration
  eta = 3 # defines downsampling rate (default=3)
  logeta = lambda x: np.log(x)/np.log(eta)
  s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
  B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

  #### Begin Finite Horizon Hyperband outerloop. Repeat indefinetely.
  for s in reversed(range(s_max+1)):

    n = int(np.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
    r = max_iter*eta**(-s) # initial number of iterations to run configurations for

    #### Begin Finite Horizon Successive Halving with (n,r)
    
    T = [ get_random_hyperparameterset(config) for i in range(n) ]
    #print T
    for i in range(s+1):
      # Run each of the n_i configs for r_i iterations and keep best n_i/eta
      n_i = n*eta**(-i)
      r_i = r*eta**(i)
      print '######################################'
      print 'keep best: ', n_i/eta
      print 'number of epochs: ', int(r_i)
      print 'number of configs: ', len(T)
      print '######################################'
      #for t in T:
      #  print t
      val_losses = [ run_nn(int(r_i), t, X_train, Y_train) for t in T ]
      T = [ T[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
    
    print np.argsort(val_losses)[0:int( n_i/eta )]
    history_list.append(T)
    print history_list
  
  
  filename = str(config['bestparams'])
  write_params(T[0], filename)
  print np.argsort(val_losses)[0:int( n_i/eta )]
  print history_list
  
  print '> hyperparameter tuning took : ', time.time() - start
    
