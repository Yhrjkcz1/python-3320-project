import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. 读取数据
df = pd.read_csv('cleaned2_Viral_Social_Media_Trends.csv')

# 2. 计算 Engagement_Level
def calculate_engagement_level(row):
    interaction_rate = (row['Likes'] + row['Shares'] + row['Comments']) / row['Views']
    if interaction_rate > 0.1:
        return 'High'
    elif interaction_rate > 0.05:
        return 'Medium'
    else:
        return 'Low'

# 应用公式创建 Engagement_Level 列
df['Engagement_Level'] = df.apply(calculate_engagement_level, axis=1)

# 3. 编码输入特征
categorical_cols = ['Hashtag', 'Content_Type', 'Region', 'Platform']  # 加入 Platform 作为特征
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# 合并数值特征和编码特征
numerical_features = ['Views', 'Likes', 'Shares', 'Comments']  # 移除 Engagement_Score
X = pd.concat([df[numerical_features], encoded_df], axis=1)

# 4. 标签编码（Engagement_Level）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Engagement_Level'])

# 5. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 构建神经网络模型
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # 3 类：High, Medium, Low
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. 训练
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 8. 预测与评估
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))