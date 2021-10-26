import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv('', low_memory = False)
df.iloc[:,78] = df.iloc[:,78].map(lambda x : 1 if x == "BENIGN" else 0)  # iloc 78 為 Label 所在 column, 把 Label 為 BENIGN 的設為0  其餘為1
