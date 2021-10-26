from adaptive_xgboost import AdaptiveXGBoostClassifier

from skmultiflow.data import FileStream
from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential

from skmultiflow.meta import AdaptiveRandomForestClassifier

import pickle
import os 
import time


with open('AXGBr.pickle', 'rb') as f:
	AXGBr = pickle.load(f) #讀取訓練完成後的模型


import pandas as pd

df = pd.read_csv('cic2017NoLabel_3.csv') #讀取testing data



dfvalue = df.values # 將dataframe轉換成list




from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


import numpy

clabel = pd.read_csv('cic2017_3_label.csv') # 讀取ground truth來做後續比較



clabelV = clabel.T.values #根據讀取的資料做轉置，不一定需要 transpose
clabelV = clabelV[0]

# print(clabelV)



start_time = time.time() #用來記錄prediction所花的時間
temp = ARF.predict(dfvalue)
end_time = time.time()
print(temp)





print('result')
print("time cost: ",(end_time-start_time))
print('confusion_matrix :\n', confusion_matrix(clabelV, temp)) #根據自己需要的metric做變化，
print('accuracy_score :', accuracy_score(clabelV, temp))
print('recall_score :', recall_score(clabelV, temp))
print('precision_score :', precision_score(clabelV, temp))
print('f1_score :', f1_score(clabelV, temp))

