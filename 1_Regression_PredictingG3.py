import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu
df = pd.read_csv(r'c:\Users\Admin\pthon\week7\student-mat.csv', sep=';')

print("Tải dữ liệu thành công!")

# CHỈ DÙNG CỘT SỐ
X = df[['G1', 'G2', 'studytime', 'failures']]
y = df['G3']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()

print("Đang train...")
model.fit(X_train, y_train)
print("Train xong")