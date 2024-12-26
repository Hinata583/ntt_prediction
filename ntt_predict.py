import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime

#データの読み込み
data = pd.read_csv("C:\\Users\\56hin\\OneDrive\\プログラム\\stocks_pile.csv")
data["date"]=pd.to_datetime(data["date"])
data=data.sort_values('date',ascending=True)
data.set_index("date",inplace=True)
data = data[["close"]]
#正規化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
#データ分割
training_size = int(len(data_scaled) * 0.6)
train_data, test_data = data_scaled[:training_size], data_scaled[training_size:]

# データをLSTM用に変換
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# LSTMの入力の形状を (samples, time_steps, features) に変換
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# モデルのコンパイルと訓練
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)
# 予測
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 予測結果を逆変換
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# モデルの評価、RMSEとR-squared
mse = mean_squared_error(y_test[0], test_predict[:, 0])
rmse = mse**0.5
r2 = r2_score(y_test[0], test_predict[:, 0])

print('RMSE: {:.2f}'.format(rmse))
print('R^2: {:.2f}'.format(r2))

# プロット
#　ｘ軸日付の反映
fig, ax = plt.subplots()
locator=mdates.YearLocator()
ax.xaxis.set_major_locator(locator)
formatter=mdates.DateFormatter("%Y")
ax.xaxis.set_major_formatter(formatter)
#元データのプロットのプロット
ax.plot(data.index, data, label='Historical')
fig.autofmt_xdate(rotation=90,ha="center")
#学習データに対する予測のプロット
train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
ax.plot(data.index, train_predict_plot, label='Train Predict')
#テストデータに対する予測のプロット
fig.autofmt_xdate(rotation=90,ha="center")
test_predict_plot = np.empty_like(data_scaled)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1, :] = test_predict
ax.plot(data.index, test_predict_plot, label='Test Predict')
fig.autofmt_xdate(rotation=90,ha="center")
plt.legend()
plt.show()
