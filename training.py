from adaptive_xgboost import AdaptiveXGBoostClassifier


from skmultiflow.data import FileStream
from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential



from skmultiflow.meta import AdaptiveRandomForestClassifier
import pickle

# import matplotlib
# matplotlib.use('TkAgg')

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.15    # Learning rate or eta
max_depth = 10          # Max depth for each tree in the ensemble
max_window_size = 500  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection


AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)


# arf = AdaptiveRandomForestClassifier(max_features='30') #測試用的baseline algo, implemented in skmultiflow lib.


################# training ##############


stream = FileStream('cicIDS2017_1.csv') # 讀取training data,會抓取最後一個column當作label

# stream.prepare_for_use()   # Required for skmultiflow v0.4.1 ，這步驟非必要

evaluator = EvaluatePrequential(pretrain_size=0, 
                                max_samples=230901,# 要自行設定訓練到第幾筆資料,預設只會訓練十萬筆資料
                                show_plot=False, 
                                metrics=['accuracy', 'kappa', 'f1', 'recall', 'running_time']
                                )#


evaluator.evaluate(stream=stream,
                   model=[AXGBr], #在這邊決定要有哪些模型要訓練，可以同時有多個模型進行
                   model_names=['model_AXGBr']) # 這邊決定每個模型的顯示名稱，在訓練結束時顯示，或是 show_plot 為 true 時在圖表上顯示的名字



################# training ##############


#若有多個csv檔案需要依次訓練，上面 ### training ### 之間的code可以重複，如以下

################# example ###############

# stream = FileStream('cicIDS2017_2.csv') # 這邊改成用另一個csv檔做訓練，會接續上一步驟訓練完的模型繼續訓練

# # stream.prepare_for_use()   # Required for skmultiflow v0.4.1， 非必要

# evaluator = EvaluatePrequential(pretrain_size=0, 
#                                 max_samples=230901,
#                                 show_plot=False, 
#                                 metrics=['accuracy', 'kappa', 'f1', 'recall', 'running_time']
#                                 )#


# evaluator.evaluate(stream=stream,
#                    model=[AXGBr], 
#                    model_names=['model_AXGBr']) 



with open('AXGBr.pickle', 'wb') as f:
    pickle.dump(AXGBr, f) #訓練完成後將模型資訊dump出來


# with open('ARF_1.pickle', 'wb') as f:
#     pickle.dump(arf, f)