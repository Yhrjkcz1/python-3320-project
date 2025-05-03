import pandas as pd

# 读取数据
df = pd.read_csv('Viral_Social_Media_Trends.csv')  # 如果你是继续用上面的路径，则不需要改

# 查看基本信息
print(df.info())

# 先删除不需要的列，比如 Post_ID
df = df.drop(columns=['Post_ID'])

# 把 Engagement_Level 转为数值（Low=0, Medium=1, High=2）
engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Engagement_Level'] = df['Engagement_Level'].map(engagement_map)

# 创建一个新的 Engagement_Score（可选）
df['Engagement_Score'] = (df['Likes'] + df['Shares'] + df['Comments']) / (df['Views'] + 1)  # 防止除以0

# 显示前几行看看结果
print(df.head())

from sklearn.preprocessing import OneHotEncoder

# 选择要编码的列
categorical_cols = ['Hashtag', 'Content_Type', 'Region']

# 创建 OneHotEncoder 实例
encoder = OneHotEncoder(sparse_output=False)

# 拟合并转换
encoded_array = encoder.fit_transform(df[categorical_cols])

# 将编码后的数据转成 DataFrame，并加上列名
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# 将原来的分类列删掉，再加上编码列
df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# 查看编码后的数据前几行
print(df_encoded.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 数值特征
numerical_features = ['Views', 'Likes', 'Shares', 'Comments', 'Engagement_Score', 'Engagement_Level']

# 找出原来编码后的分类特征（即除了 Platform、Engagement 等数值列之外的）
categorical_encoded_features = [col for col in df_encoded.columns 
                                if col.startswith('Hashtag_') or col.startswith('Content_Type_') or col.startswith('Region_')]

# 合并所有特征
all_features = numerical_features + categorical_encoded_features

# 定义特征和目标
X = df_encoded[all_features]
y = df_encoded['Platform']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 重新训练模型
model = RandomForestClassifier(random_state=42, class_weight='balanced')  # 加入 class_weight 对付类别不均衡
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

print(df['Platform'].value_counts(normalize=True))

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# 假设 X 和 y 是前面处理好的（包含编码和数值特征）
# 如果你是接着上一步继续的，X、y 不需要变

# 使用 SMOTE 过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("过采样后类别分布：")
print(pd.Series(y_resampled).value_counts())

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("新准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
