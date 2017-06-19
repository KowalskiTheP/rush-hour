from ConfigParser import SafeConfigParser
import sys

import pprint

def readINI(filename):
  """reads the ini file and checks if files and prerequisites are met"""
  #actual parsing
  confParser = SafeConfigParser()
  confParser.read(filename)

  configfileArgs={}

  for section_name in confParser.sections():
    for name, value in confParser.items(section_name):
      configfileArgs[name] = confParser.get(section_name, name)
  
  #print configfileArgs
  #print confParser.sections()
  for name, value in confParser.items('input'):
    if ',' in configfileArgs[name]:
      configfileArgs[name] = value.split(',')
  
  if isinstance(configfileArgs['columns'], list):
    valuesNew = [int(item) for item in configfileArgs['columns']]
    configfileArgs['columns'] = valuesNew
  else:
    configfileArgs['columns'] = [int(configfileArgs['columns'])]
    
  configfileArgs['y_column'] = int(configfileArgs['y_column'])

  
  for name, value in confParser.items('network'):
    if ',' in configfileArgs[name]:
      configfileArgs[name] = value.split(',')

  for name, value in confParser.items('tuning'):
    if ',' in configfileArgs[name]:
      configfileArgs[name] = value.split(',')
  #print configfileArgs
  
  if isinstance(configfileArgs['neuronsperlayer'], list) == isinstance(configfileArgs['activationperlayer'], list) and isinstance    (configfileArgs['neuronsperlayer'], list) == True:
    if len(configfileArgs['neuronsperlayer']) != len(configfileArgs['activationperlayer']):
      print 'number of activation functions has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    if  isinstance(configfileArgs['neuronsperlayer'], list) == True:
      temp_list1 = []
      for item in configfileArgs['neuronsperlayer']:
        temp_list1.append(configfileArgs['activationperlayer'])
      configfileArgs['activationperlayer'] = temp_list1

  if isinstance(configfileArgs['neuronsperlayer'], list) == isinstance(configfileArgs['recurrentactivation'], list) and isinstance    (configfileArgs['neuronsperlayer'], list) == True:
    if len(configfileArgs['neuronsperlayer']) != len(configfileArgs['recurrentactivation']):
      print 'number of recurrent activation functions has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    if  isinstance(configfileArgs['neuronsperlayer'], list) == True:    
      temp_list1 = []
      for item in configfileArgs['neuronsperlayer']:
        temp_list1.append(configfileArgs['recurrentactivation'])
      configfileArgs['recurrentactivation'] = temp_list1    
  
  if isinstance(configfileArgs['neuronsperlayer'], list) == isinstance(configfileArgs['dropout'], list) and isinstance    (configfileArgs['neuronsperlayer'], list) == True:
    if len(configfileArgs['neuronsperlayer']) != len(configfileArgs['dropout']):
      print 'number of dropout has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
   if  isinstance(configfileArgs['neuronsperlayer'], list) == True:    
     temp_list1 = []
     for item in configfileArgs['neuronsperlayer']:
       temp_list1.append(configfileArgs['dropout'])
     configfileArgs['dropout'] = temp_list1
  
  pp = pprint.PrettyPrinter(indent=2)
  pp.pprint(configfileArgs)

      

 
  for name, value in confParser.items('tuning'):
    if configfileArgs[name] is not None:
      if name == 'nlayer_tune':
        valuesNew = [int(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew
      elif name == 'actlayer_tune':
        valuesNew = [str(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew
      elif name == 'nhiduplayer_tune':
        valuesNew = [int(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew
      elif name == 'dropout_tune':
        valuesNew = [float(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew
      elif name == 'lr_tune':
        valuesNew = [float(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew
      elif name == 'batchsize_tune':
        valuesNew = [int(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew
      elif name == 'batchnorm_tune':
        valuesNew = [str(item) for item in configfileArgs[name]]
        configfileArgs[name] = valuesNew

  return configfileArgs
