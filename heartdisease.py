from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## 資料讀取
df = pd.read_csv("F:\PythonPortfolio\project_C\heart.csv")

# Age           患者年齡[歲]
# Sex           患者性別[M=男, F=女]
# ChestPainType 胸痛類型【TA=典型心絞痛, ATA=非典型心絞痛, NAP=非心絞痛, ASY=無症狀】
# RestingBP     靜息血壓 [mm Hg]
# Cholesterol   血清膽固醇 [mm/dl]
# FastingBS     空腹血糖 [if FastingBS > 120 mg/dl, else 0]
# RestingECG    靜息心電圖結果[正常：正常，ST：有 ST-T 波異常（T 波倒置和/或 ST 抬高或壓低 > 0.05 mV），LVH：根據 Estes 標準顯示可能或明確的左心室肥厚]
# MaxHR         達到的最大心率 [60 到 202 之間的數值]
# ExerciseAngina運動誘發的心絞痛[Y：是，N：否]
# Oldpeak       ST [抑鬱症測量的數值]
# ST_Slope      峰值運動ST段的斜率[Up：向上傾斜，Flat：平坦，Down：向下傾斜]
# HeartDisease  輸出類[1：心髒病，0：正常]


print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())
print(df.columns)
print(df.isna().sum())


## 資料前處理
## 視覺化
plt.style.use("bmh")
plt.figure(figsize=(8, 6))
# feature = df.drop("HeartDisease", axis=1)
# for i, feature in enumerate(feature.columns):
#     plt.subplots_adjust(hspace=0.8)
#     plt.subplot(6, 2, i+1)
#     sns.histplot(data=df, x=feature, kde=True,
#                  hue="HeartDisease")


# 年齡 ( Age ) : 性別與年齡 小提琴圖 圓餅圖 ( hue = HeartDisease )
plt.subplot(1, 2, 1)
sns.violinplot(data=df, x="Sex", y="Age", hue="HeartDisease", split=True)
plt.subplot(1, 2, 2)
df.groupby('Sex').size().plot(kind='pie', autopct='%.0f%%')
plt.title("Sex")
plt.show()
# 胸痛類型（ChestPainType）：類型圓餅圖比例, 長條圖 ( hue = HeartDisease )
plt.figure(figsize=(10, 6))
plt.title("ChestPainType")
plt.subplot(1, 2, 1)
df.groupby("ChestPainType").size().plot(kind='pie', autopct='%.0f%%')
plt.subplot(1, 2, 2)
sns.countplot(data=df, x="ChestPainType", hue="HeartDisease")
plt.show()
# 靜息心電圖（RestingECG）：長條圖 ( hue = HeartDisease )
plt.title("RestingECG")
sns.countplot(data=df, x="RestingECG", hue="HeartDisease")
plt.show()
# 血清膽固醇（Cholesterol）： 膽固醇與年紀 盒狀圖 ( hue = HeartDisease )
chol = df["Cholesterol"].apply(
    lambda x: "VeryHigh" if x >= 240 else ("High" if x >= 200 else "Normal"))
sns.boxplot(x=chol, y=df["Age"], hue=df["HeartDisease"])
sns.swarmplot(y="Age", x=chol, data=df, color="r")
plt.title("Cholesterol")
plt.show()
# 達到的最大心率（MaxHR）： 年齡與最大心率 散佈圖 ( hue = HeartDisease )
sns.lmplot(data=df, x="Age", y="MaxHR", hue="HeartDisease", col="Sex")
plt.show()
# 運動誘發的心絞痛（ExerciseAngina）：可以使用計數圖或餅圖顯示運動誘發的心絞痛的分佈情況
# ，並將其與患心臟病的人進行對比。
plt.subplot(1, 2, 1)
df.groupby("ExerciseAngina").size().plot(kind="pie", autopct='%.0f%%')
plt.subplot(1, 2, 2)
plt.title("ExerciseAngina")
sns.countplot(data=df, x="ExerciseAngina", hue="HeartDisease")
plt.show()
# ST_Slope 峰值運動ST段的斜率
sns.histplot(data=df, x="ST_Slope", hue="HeartDisease", multiple="dodge")
plt.show()
# RestingBP 靜息血壓
# 數據異常異常(血壓為0)
sns.lmplot(data=df, x="Age", y="RestingBP", hue="HeartDisease")
plt.show()
print(df[df["RestingBP"] == 0])
# 數據處理
RestingBP_mean = df["RestingBP"].replace(0, np.nan).mean()
print(RestingBP_mean)
df["RestingBP"] = df["RestingBP"].replace(0, RestingBP_mean)
print(df[df["RestingBP"] == 0])
sns.lmplot(data=df, x="Age", y="RestingBP", hue="HeartDisease")
plt.show()


## 特徵工程
# Sex 患者性別[M=男, F=女]
df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "M" else 0)
print(df["Sex"].value_counts())

# ST_Slope 峰值運動ST段的斜率[Up：向上傾斜，Flat：平坦，Down：向下傾斜]
def st_slope_num(x):
    if x == "Up":
        return 1
    elif x == "Flat":
        return 2
    else:
        return 3
df["ST_Slope"] = df["ST_Slope"].apply(st_slope_num)
print(df["ST_Slope"].value_counts())

# ChestPainType 胸痛類型【TA=典型心絞痛, ATA=非典型心絞痛, NAP=非心絞痛, ASY=無症狀】
def cptype(x):
    if x == "TA":
        return 1
    elif x == "ATA":
        return 2
    elif x == "NAP":
        return 3
    else:
        return 4
df["ChestPainType"] = df["ChestPainType"].apply(cptype)
print(df["ChestPainType"].value_counts())

# ExerciseAngina 運動誘發的心絞痛[Y：是，N：否]
df["ExerciseAngina"] = df["ExerciseAngina"].apply(
    lambda x: 1 if x == "Y" else 0)
print(df["ExerciseAngina"].value_counts())
# RestingECG 靜息心電圖結果
# normal：正常
# ST：有 ST-T 波異常（T 波倒置和/或 ST 抬高或壓低 > 0.05 mV)
# LVH：根據 Estes 標準顯示可能或明確的左心室肥厚
df["RestingECG"] = df["RestingECG"].apply(
    lambda x: 0 if x == "Normal" else (1 if x == "LVH" else 2))
print(df["RestingECG"].value_counts())


# 各特徵相關熱力圖
df_corr = df.corr()
print(df_corr["HeartDisease"].sort_values(ascending=False))
sns.heatmap(df_corr, annot=True, fmt=".2f")
plt.show()

# 刪除最不相關的前五個特徵
print(df_corr["HeartDisease"].sort_values(ascending=True))
top_feature = df.drop(
    ["FastingBS", "RestingBP", "RestingECG", "Cholesterol", "MaxHR"], axis=1)
print(top_feature.columns)
sns.pairplot(top_feature, kind="reg", diag_kind="auto", hue="HeartDisease")
plt.show()


# %%   Modeling


X = df.drop(["HeartDisease"], axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% 隨機森林

rfc = RandomForestClassifier(random_state=20)

# # GridSearchCV 參數最佳化
param_grid = {
    "n_estimators": list(range(20, 100, 10)),
    "max_depth": list(range(2, 9, 1))
}
rfc_grid_cv = GridSearchCV(rfc, param_grid, cv=5)

# # 訓練及預測
rfc_grid_cv.fit(X_train, y_train)
print(
    f"best_estimator: {rfc_grid_cv.best_estimator_}\n\
    best_score: {rfc_grid_cv.best_score_:.2f}")
y_rfc_pred = rfc_grid_cv.predict(X_test)
print(f"預測結果: \n{y_rfc_pred}")

# # 準確率
accuracy = metrics.accuracy_score(y_test, y_rfc_pred)
print(f"rfc_Accuracy: : {accuracy:.3f}")

# # 混淆矩陣
cm = confusion_matrix(y_test, y_rfc_pred)
print("Matrix: \n", cm)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[True, False])
cm_display.plot()
plt.show()


# %% SVM 標準化預測

# # 建立 Pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                ('svc', SVC())])

# # GridSearchCV 參數最佳化
param_grid = {
    "svc__C": np.linspace(1, 1.5, 10),
    "svc__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "svc__gamma": ['scale', 'auto']
}
svc_grid_cv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)

# # 訓練及預測
svc_grid_cv.fit(X_train, y_train)
print(
    f"best_params: {svc_grid_cv.best_params_}\n\
    best_score: {svc_grid_cv.best_score_:.2f}")
y_svc_pred = svc_grid_cv.predict(X_test)
print(f"預測結果: \n{y_svc_pred}")

# # 準確率
accuracy = metrics.accuracy_score(y_test, y_svc_pred)
print(f"SVC_Accuracy: {accuracy:.3f}")

# # 混淆矩陣
cm = confusion_matrix(y_test, y_svc_pred)
print("Matrix: \n", cm)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[True, False])
cm_display.plot()
plt.show()


# %% SVM 沒有標準化預測

# svc = SVC(random_state=20, C=1.5, kernel="rbf", gamma="scale")

# # # 訓練及預測
# svc.fit(X_train, y_train)
# y_svc_pred = svc.predict(X_test)
# print(f"預測結果: \n{y_svc_pred}")

# # # 準確率
# accuracy = metrics.accuracy_score(y_test, y_svc_pred)
# print(f"SVC_Accuracy: {accuracy:.3f}")

# # # 混淆矩陣
# cm = confusion_matrix(y_test, y_svc_pred)
# print("Matrix: \n", cm)
# cm_display = metrics.ConfusionMatrixDisplay(
#     confusion_matrix=cm, display_labels=[True, False])
# cm_display.plot()
# plt.show()


# %% 邏輯回歸

# # 邏輯回歸 標準化
lr = LogisticRegression(random_state=20)

# # 建立Pipeline 標準化與模型
pipe = Pipeline([("scaler", StandardScaler()),
                 ("lr", lr)])

# # 訓練及預測
pipe.fit(X_train, y_train)
y_lr_pred = pipe.predict(X_test)
print(f"預測結果: \n{y_lr_pred}")

# # 準確率
accuracy = metrics.accuracy_score(y_test, y_lr_pred)
print(f"lr_Accuracy: {accuracy:.3f}")

# # 混淆矩陣
cm = confusion_matrix(y_test, y_lr_pred)
print("Matrix: \n", cm)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[True, False])
cm_display.plot()
plt.show()


# %% 邏輯回歸 沒有標準化預測

# lr = LogisticRegression(max_iter=1000, random_state=20)

# # # 訓練及預測
# lr.fit(X_train, y_train)
# y_lr_pred = lr.predict(X_test)
# print(f"預測結果: \n{y_lr_pred}")

# # # 準確率
# accuracy = metrics.accuracy_score(y_test, y_lr_pred)
# print(f"lr_Accuracy: {accuracy:.4f}")

# # 混淆矩陣
# cm = confusion_matrix(y_test, y_lr_pred)
# print("Matrix: \n", cm)
# cm_display = metrics.ConfusionMatrixDisplay(
#     confusion_matrix=cm, display_labels=[True, False])
# cm_display.plot()
# plt.show()

# %%
