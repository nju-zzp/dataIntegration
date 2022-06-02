from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

# 1、获取数据
raw = pd.read_csv("./trading.csv")
trading = raw[raw['credit_level'] != -1]
trading_test = raw[raw['credit_level'] == -1]
uid_credit_raw = pd.read_csv("./uid_credit.csv")
uid_list = (uid_credit_raw[uid_credit_raw["credit_level"] == -1])['uid'].values

trading_data = trading[['etc_amt', 'sa_amt', 'sbyb_amt', 'sjyh_amt']].values
trading_target = trading['credit_level'].values

# 2、划分训练集和测试集   注意顺序不能乱分别表示 训练集，测试集，训练集标签（类别），测试集标签（类别）
Xtrain, Xtest, Ytrain, Ytest = train_test_split(trading_data, trading_target, test_size=0.2)


# 3、训练、测试与预测
clf = tree.DecisionTreeClassifier()
clf.fit(Xtrain,Ytrain)               #拟合
score = clf.score(Xtest,Ytest)         #模型评分
prediction = clf.predict(trading_test[['etc_amt', 'sa_amt', 'sbyb_amt', 'sjyh_amt']].values)

result = list(zip(uid_list, prediction))
result = pd.DataFrame(result, columns=['uid', 'credit'])

print(score)
print(list(prediction))

result.to_csv("./18_credit.csv")