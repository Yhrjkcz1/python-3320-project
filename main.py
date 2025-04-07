import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
df = pd.read_csv('Viral_Social_Media_Trends.csv')

# 数据预处理：删除缺失值
df = df.dropna()

# 定义特征和目标变量
X = df.drop('Engagement_Level', axis=1)
y = df['Engagement_Level']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建预处理管道（处理分类变量）
categorical_features = ['Platform', 'Content_Type', 'Region']
numeric_features = ['Views', 'Likes', 'Shares', 'Comments']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 构建机器学习管道
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 模型评估
y_pred = pipeline.predict(X_test)
print("模型准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 新帖子预测示例
new_post = pd.DataFrame({
    'Platform': ['TikTok'],
    'Content_Type': ['Video'],
    'Region': ['UK'],
    'Views': [4000000],
    'Likes': [300000],
    'Shares': [50000],
    'Comments': [20000]
})

prediction = pipeline.predict(new_post)
print("\n新帖子参与度预测:", prediction[0])