from ConfigParser import SafeConfigParser
import sys
from bunch import bunchify, unbunchify

import pprint

def readINI(filename):
  """reads the ini file and checks if files and prerequisites are met"""
  #actual parsing
  confParser=SafeConfigParser()
  confParser.read(filename)

  conf={}

  for section_name in confParser.sections():
    for name, value in confParser.items(section_name):
      conf[name]=confParser.get(section_name, name)
  
  #print conf
  #print confParser.sections()
  for name, value in confParser.items('input'):
    if ',' in conf[name]:
      conf[name]=value.split(',')
  
  if isinstance(conf['columns'], list):
    valuesNew=[int(item) for item in conf['columns']]
    conf['columns']=valuesNew
  else:
    conf['columns']=[int(conf['columns'])]
    
  conf['y_column']=int(conf['y_column'])

  
  for name, value in confParser.items('network'):
    if ',' in conf[name]:
      conf[name]=value.split(',')

  for name, value in confParser.items('tuning'):
    if ',' in conf[name]:
      conf[name]=value.split(',')
  #print conf
  print 'neuronen PRO LAYER',conf['neuronsperlayer']
  if isinstance(conf['neuronsperlayer'], list) == isinstance(conf['activationperlayer'], list) and isinstance(conf['neuronsperlayer'], list) == True:
    if len(conf['neuronsperlayer']) != len(conf['activationperlayer']):
      print 'number of activation functions has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    if  isinstance(conf['neuronsperlayer'], list) == True:
      temp_list1=[]
      for item in conf['neuronsperlayer']:
        temp_list1.append(str(conf['activationperlayer']))
      conf['activationperlayer']=temp_list1

  if isinstance(conf['neuronsperlayer'], list) == isinstance(conf['recurrentactivation'], list) and isinstance(conf['neuronsperlayer'], list) == True:
    if len(conf['neuronsperlayer']) != len(conf['recurrentactivation']):
      print 'number of recurrent activation functions has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    if  isinstance(conf['neuronsperlayer'], list) == True:    
      temp_list1=[]
      for item in conf['neuronsperlayer']:
        temp_list1.append(str(conf['recurrentactivation']))
      conf['recurrentactivation']=temp_list1    
  
  if isinstance(conf['neuronsperlayer'], list) == isinstance(conf['dropout'], list) and isinstance(conf['neuronsperlayer'], list) == True:
    if len(conf['neuronsperlayer']) != len(conf['dropout']):
      print 'number of dropout has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
   if  isinstance(conf['neuronsperlayer'], list) == True:    
     temp_list1=[]
     for item in conf['neuronsperlayer']:
       temp_list1.append(float(conf['dropout']))
     conf['dropout']=temp_list1

  if  isinstance(conf['neuronsperlayer'], list) == True:
    valuesNew=[int(item) for item in conf['neuronsperlayer']]
    conf['neuronsperlayer']=valuesNew
  else:
    conf['neuronsperlayer']=int(conf['neuronsperlayer'])

  if  isinstance(conf['columns'], list) == True:
    valuesNew=[int(item) for item in conf['columns']]
    conf['columns']=valuesNew
  else:
    conf['columns']=int(conf['columns'])
 
  for name, value in confParser.items('tuning'):
    if conf[name] is not None:
      if name == 'nlayer_tune':
        valuesNew=[int(item) for item in conf[name]]
        conf[name]=valuesNew
      elif name == 'actlayer_tune':
        valuesNew=[str(item) for item in conf[name]]
        conf[name]=valuesNew
      elif name == 'nhiduplayer_tune':
        valuesNew=[int(item) for item in conf[name]]
        conf[name]=valuesNew
      elif name == 'dropout_tune':
        valuesNew=[float(item) for item in conf[name]]
        conf[name]=valuesNew
      elif name == 'lr_tune':
        valuesNew=[float(item) for item in conf[name]]
        conf[name]=valuesNew
      elif name == 'batchsize_tune':
        valuesNew=[int(item) for item in conf[name]]
        conf[name]=valuesNew
      elif name == 'batchnorm_tune':
        valuesNew=[str(item) for item in conf[name]]
        conf[name]=valuesNew
  
  # bunchify the configfile arguments, access like java struct e.g. conf.neuronsperlayer
  conf=bunchify(conf)
  
  '''setting the right datatypes for the configfile arguments
  ATTENTION!!!! IF NEW configfile arguments gets added it must be inserted here!!!!
  config file arguments that are list get handled above!!'''

  conf.attention=str(conf.attention)
  conf.batchnorm=str(conf.batchnorm)
  conf.batchsize=int(conf.batchsize)
  conf.bestparams=str(conf.bestparams)
  conf.bidirect=str(conf.bidirect)
  conf.cnn=str(conf.cnn)
  conf.csvfile=str(conf.csvfile)
  conf.datecolumn=str(conf.datecolumn)
  conf.datetopred=str(conf.datetopred)
  conf.decay=float(conf.decay)
  conf.earlystop=float(conf.earlystop)
  conf.epochs=int(conf.epochs)
  conf.evalmetrics=str(conf.evalmetrics)
  conf.header=int(conf.header)
  conf.initweights=str(conf.initweights)
  conf.inputdim=int(conf.inputdim)
  conf.jsonfile=str(conf.jsonfile)
  conf.learningrate=float(conf.learningrate)
  conf.loglevel=int(conf.loglevel)
  conf.look_back=int(conf.look_back)
  conf.loss=str(conf.loss)
  conf.modelfile= str(conf.modelfile)
  conf.normalise=int(conf.normalise)
  conf.optimiser=str(conf.optimiser)
  conf.outputdim=int(conf.outputdim)
  conf.outputlength=int(conf.outputlength)
  conf.plotting=str(conf.plotting)
  conf.predictionfile=str(conf.predictionfile)
  conf.refvalue=float(conf.refvalue)
  conf.smoothingswitch=str(conf.smoothingswitch)
  conf.smoothingparam=int(conf.smoothingparam)
  conf.split=str(conf.split)
  conf.timedistributed=str(conf.timedistributed)
  conf.traintestsplit=float(conf.traintestsplit)
  conf.tuning=str(conf.tuning)
  conf.verbosity=str(conf.verbosity)
  conf.winlength=int(conf.winlength)
  conf.y_column=int(conf.y_column)
  
  pp=pprint.PrettyPrinter(indent=2)
  pp.pprint(unbunchify(conf))
  
  return conf
