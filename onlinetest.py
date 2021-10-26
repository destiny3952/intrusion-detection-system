import os 
import numpy, collections, time

#do tcpdump to capture packets
print(os.system("sudo tcpdump -c 5000 -w demo.pcap")) # 透過 tcpdump 抓取 network traffic，存到 pcap file 中


#use argus and ra to transform pcap file into flow feature based csv file

start_time = time.time() # 觀察 extract feature 所需要的時間
print(os.system("sudo argus -r demo.pcap -w demo.argus")) # 需要先把 pcap file 轉換成 argus file，後續才能用 ra 把flow feature抓出來
print(os.system("ra -r demo.argus -u -s saddr,sport,daddr,dport,proto,dur,sbytes,dbytes,sttl,dttl,sloss,dloss,sload,dload,spkts,dpkts,swin,dwin,stcpb,dtcpb,smeansz,dmeansz,sjit,djit,stime,ltime,sintpkt,dintpkt,tcprtt,synack,ackdat -c , > test2.csv"))
#把extract出feature的資料存到test2.csv檔案中
end_time = time.time()
print("argus time cost: ",(end_time-start_time))


from adaptive_xgboost import AdaptiveXGBoostClassifier

# from skmultiflow.data import FileStream
# from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential



from skmultiflow.meta import AdaptiveRandomForestClassifier

# from skmultiflow.trees import HoeffdingTreeClassifier
# from skmultiflow.bayes import NaiveBayes
# from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier



#can read after ra outputs file directly


#load model
import pickle

with open('AXGBr.pickle', 'rb') as f:
	AXGBr = pickle.load(f) # 讀取訓練好的model






#load input csv file and preprocessing
import pandas as pd

start_time = time.time() #timer of load data and preprocessing

df = pd.read_csv('test2.csv') #flow feature base csv file
dftest = pd.read_csv('unswnb15_form.csv') #one row of unsw-nb15 dataset used as form

#change column name to fit with unsw nb15
df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat']

#drop some columns
df = df.drop(columns = ['srcip', 'dstip', 'Stime', 'Ltime'])


#one hot encoding
dfdummies = pd.get_dummies(df, columns = ['proto'])

#delete proto_man column
dfdummies = dfdummies.drop(columns = ['proto_man'])


#change to unswnb15 like form
dftemp = pd.concat([dftest, dfdummies], ignore_index = True)

#delete record start, record end and form row
dftemp = dftemp.iloc[2:-1]

#drop columns
dftemp = dftemp.drop(columns = ['sport', 'dsport', 'Label'])


#fill non value col
dftemp = dftemp.fillna(0)



# df = df.drop(columns = ['Label'])

#change dataframe into list
dfvalue = dftemp.values

end_time = time.time()
print("load data and preprocessing time cost: ",(end_time-start_time))




numpy.set_printoptions(precision = 2)


#predict

start_time = time.time()
temp = AXGBr.predict(dfvalue)
# temp = temp.round(2)
end_time = time.time()




# print(temp)
# print(collections.Counter(temp))
# count = collections.Counter(temp)
# print("normal traffic count", count[0])
# print("abnormal traffic detected count: ", count[1])

#get index of abnormal traffic
mindex = [i for i, e in enumerate(temp) if e == 1]

print("abnormal traffic index: ", mindex) # 印出被預測為惡意流量的index，可以以此 index 從 input file csv 看是哪個 IP 發出來

print("time cost: ",(end_time-start_time)) # 預測時所花的時間
numpy.savetxt("predictfile/predict_argus_test_AXGBr_online.csv", temp, fmt = '%d') # 導出預測結果
