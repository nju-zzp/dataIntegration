from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

# 1、获取数据
raw = pd.read_csv("./star.csv")
star = raw[raw['star_level'] != -1]
star_test = raw[raw['star_level'] == -1]

star_data = star[['sa_bal', 'td_bal', 'fin_bal']].values
star_target = star['star_level'].values

# 2、划分训练集和测试集   注意顺序不能乱分别表示 训练集，测试集，训练集标签（类别），测试集标签（类别）
Xtrain, Xtest, Ytrain, Ytest = train_test_split(star_data, star_target, test_size=0.2)


# 3、训练、测试与预测
clf = tree.DecisionTreeClassifier()
clf.fit(Xtrain, Ytrain)               #拟合
score = clf.score(Xtest, Ytest)         #模型评分
prediction = clf.predict(star_test[['sa_bal', 'td_bal', 'fin_bal']].values)
print(score)
print(prediction)