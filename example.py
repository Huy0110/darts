import pandas as pd
from darts import TimeSeries

# Read a pandas DataFrame
df = pd.read_csv("datasets/BTCUSDT.csv", delimiter=",")

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "OPEN_TIME", "VOLUME" ,fill_missing_dates=True, freq=None)

# Set aside the last 36 months as a validation series
train, val = series[:-1000], series[-1000:]

from darts.models import NLinearModel

# Tạo mô hình TFT
model = NLinearModel(input_chunk_length=100, output_chunk_length=50, n_epochs = 20)


# Huấn luyện mô hình
model.fit(train)

# Dự đoán trên tập xác thực
prediction = model.predict(len(val))


import matplotlib.pyplot as plt

series[-2000: ].plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()

plt.savefig("forecast_plot.png")