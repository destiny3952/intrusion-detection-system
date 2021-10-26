# intrusion-detection-system

training.py 是訓練模型用的python script

predict_test.py 用來測試模型準確度

onlinetest.py 將訓練完的模型試用在實際網路監控，其中資料前處理的部分以UNSW-NB15為基準，所抓出的featrue以UNSW-NB15中feature的名稱做轉換

ids2017preprocess.py 中僅有處理label的部分，負責把 BENIGN 的 LABEL 轉換為 0 ，非 BENIGN 的都轉換為1，以利訓練