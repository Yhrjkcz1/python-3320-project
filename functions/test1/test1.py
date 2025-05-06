import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 读取数据
df = pd.read_csv('cleaned2_Viral_Social_Media_Trends.csv')

# 2. 编码输入特征
categorical_cols = ['Hashtag', 'Content_Type', 'Region']
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# 合并数值特征和编码特征
numerical_features = ['Views', 'Likes', 'Shares', 'Comments', 'Engagement_Score']
X = pd.concat([df[numerical_features], encoded_df], axis=1)

# 3. 标签编码（Platform）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Platform'])

# 4. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 构建神经网络模型
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # 多分类
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 训练
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 7. 预测与评估
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
