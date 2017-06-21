import pandas as pd
import numpy as np
import math
from datetime import datetime

def convertDate(df, dateColumn, date_format, refdate ):
  date_list = []
  b = datetime.strptime(refdate, date_format)
  for i in range(len(df)):
    a = datetime.strptime(str(df.loc[i,dateColumn]), date_format)
    date_list.append(str(a-b))
    date_list[-1] = [int(s) for s in date_list[-1].split() if s.isdigit() ]
  date_list = np.array(date_list).flatten()
  df['days'] = date_list   
  return df

def convertTime(df, timeColumn):
  date_list = []
  for i in range(len(df)):
    date_list.append(str(df.loc[i,timeColumn]).split(':')[0])
  date_list = np.array(date_list).flatten()
  df['time'] = date_list   
  return df

def reoder(dataframe, columns):
  dataSet = np.zeros((len(dataframe)*2, 2))
  j=0
  for i in xrange(0,len(dataframe)*2,2):
    dataSet[i,0] = dataframe.iloc[j,0]
    dataSet[i,1] = dataframe.iloc[j,columns[0]]
    dataSet[i+1,0] = dataframe.iloc[j,0]
    dataSet[i+1,1] = dataframe.iloc[j,columns[1]]
    j=j+1
  return dataSet

def addInfo(dataframe, orderedArray, finalWidth, columns2Add):
  dax_newArray = np.zeros((len(orderedArray),finalWidth))
  for i in range(0,len(orderedArray),2):
    if orderedArray[i,1] >= orderedArray[i-1,1]:
      tmp_array = np.array([ orderedArray[i,1], orderedArray[i-1,1] ])
    else:
      tmp_array = np.array([ orderedArray[i-1,1], orderedArray[i,1] ])
    dax_newArray[i]   = np.concatenate((orderedArray[i], tmp_array), axis=0)
    tmp_array_p1 = np.array(dataframe.iloc[(i/2),columns2Add])
    dax_newArray[i+1] = np.concatenate((orderedArray[i+1], tmp_array_p1), axis=0)
  return dax_newArray

#df_dax = pd.read_csv('../Data/dax_19700105_20170428.csv', decimal=',' ,sep=';', header=0)
df_dow = pd.read_csv('../Data/dj-IND_110101_170619.csv', decimal='.' ,sep=',', header=0)
#df_dax = df_dax[df_dax.High != 'null']
#df_dax = df_dax.reset_index(drop=True)

#df_nikkei = pd.read_csv('../Data/Nikkei_02051985_02052017.csv', decimal='.' ,sep=',', header=0)
#df_dowJones = pd.read_csv('../Data/DowJones_02051985_02052017.csv', decimal='.' ,sep=',', header=0)

#Fuer High low data
for i in range(len(df_dow)):
  tmpString = list(str(df_dow.loc[i,'<DATE>']))
  #print tmpString
  df_dow.loc[i,'<DATE>'] = ''.join(tmpString[0:4])+'.'+''.join(tmpString[4:6])+'.' +''.join(tmpString[6:8])
  
 
df_dow.drop(['<TICKER>','<PER>'], axis=1,inplace=True)
#df_nikkei.drop('Adj Close', axis=1,inplace=True)
#df_dowJones.drop('Adj Close', axis=1,inplace=True)

df_dow = convertDate(df_dow, '<DATE>', '%Y.%m.%d', '2010.01.01')
#df_dow = convertTime(df_dow, '<TIME>')
df_dow['<TIME>'] = df_dow['<TIME>'].multiply( 1./10000. )

print df_dow

j =0
dayLength = []
for i in range(len(df_dow)):
  if df_dow.loc[i,'<TIME>'] == 0.:
    dayLength.append( i- j )
    j = i

sieben = 0
acht = 0
neun = 0
zen = 0
elf = 0 
zw = 0 
dreizen = 0
for i in range(len(dayLength)):
  if dayLength[i] == 7: sieben+=1
  if dayLength[i] == 8: acht+=1
  if dayLength[i] == 9: neun+=1
  if dayLength[i] == 10: zen+=1
  print i
  if dayLength[i] == 11: elf+=1
  if dayLength[i] == 12: zw+=1
  if dayLength[i] == 13: dreizen+=1
  
print sieben ,acht ,neun ,zen ,elf ,zw ,dreizen 
  

#df_nikkei = convertDate(df_nikkei, 'Date', '%Y-%m-%d', '1985-01-01')
#df_dowJones = convertDate(df_dowJones, 'Date', '%Y-%m-%d', '1985-01-01')

#df_combi = pd.merge(left=df_dax, right=df_nikkei, on='days')
#df_combi = pd.merge(left=df_combi, right=df_dowJones, on='days')

df_dow = df_dow.loc[:,['days','time','Open','High','Low','Close','Volume']]

print df_dow

temp_tmp = []
for i in range(0,len(df_dow)):
  if int(df_dow.loc[i,'time']) == 10:
    temp_tmp.append([df_dow.loc[i,'days'],9,df_dow.loc[i,'Open']])
    temp_tmp.append([df_dow.loc[i,'days'],df_dow.loc[i,'time'],df_dow.loc[i,'Close']])
  else:
    temp_tmp.append([df_dow.loc[i,'days'],df_dow.loc[i,'time'],df_dow.loc[i,'Close']])
    
df_combi = pd.DataFrame(data=np.array(temp_tmp))
print df_combi

#df_nikkei = df_combi.loc[:,['days','Open_y','High_y','Low_y','Close_y']]
#df_dowJones = df_combi.loc[:,['days','Open','High','Low','Close']]

#array_dax = reoder(df_dax, [1,4])
#array_nikkei = reoder(df_nikkei, [1,4])
#array_dowJones = reoder(df_dowJones, [1,4])

#dax_newArray = addInfo(df_dax, array_dax, 4, [2,3])
#nikkei_newArray = addInfo(df_nikkei, array_nikkei, 4, [2,3])
#dowJones_newArray = addInfo(df_dowJones, array_dowJones, 4, [2,3])

#array_dax = dax_newArray
#array_nikkei = nikkei_newArray
#array_dowJones = dowJones_newArray

#for i in range(len(array_dax)):
  #if array_dax[i,0]!=array_nikkei[i,0] or array_dax[i,0]!=array_dowJones[i,0] or array_dowJones[i,0]!=array_nikkei[i,0]:
    #print 'Problem!!! Arrays are not in sync!'    

#df_dax = pd.DataFrame(data=array_dax, columns=['days','stock','high','low'])
#df_nikkei = pd.DataFrame(data=array_nikkei, columns=['days','stock','high','low'])
#df_dowJones = pd.DataFrame(data=array_dowJones, columns=['days','stock','high','low'])

#df_nikkei = df_nikkei.drop(df_nikkei.index[[0]])
#df_nikkei = df_nikkei.reset_index(drop=True)
#print 'df_nikkei:\n', df_nikkei.shift(periods=1, freq=None, axis=0)

#df_combi = pd.concat([df_dax, df_nikkei[['stock','high','low']] ], axis=1, join_axes=[df_dax.index])
#df_combi = pd.concat([df_combi, df_dowJones[['stock','high','low']] ], axis=1, join_axes=[df_combi.index])

#df_combi = df_combi.drop(df_combi.index[[12376,12377]])

df_combi.to_csv('../Data/hourly.csv',index=False)



