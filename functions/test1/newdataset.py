import pandas as pd
import matplotlib.pyplot as plt

# 读取原始 CSV 文件
df = pd.read_csv("Viral_Social_Media_Trends.csv")  # 替换为你的文件名或路径

# 删除不需要的列，比如 Post_ID
if 'Post_ID' in df.columns:
    df = df.drop(columns=['Post_ID'])

# 计算 Engagement Score：简单平均互动率
def calculate_engagement(row):
    if row['Views'] == 0:
        return 0
    return (row['Likes'] + row['Shares'] + row['Comments']) / row['Views']

# 应用计算
df['Engagement_Score'] = df.apply(calculate_engagement, axis=1)

# 根据 Engagement Score 分类 Engagement Level
def calculate_engagement_level(row):
    interaction_rate = row['Engagement_Score']
    if interaction_rate > 0.1:
        return 'High'
    elif interaction_rate > 0.05:
        return 'Medium'
    else:
        return 'Low'

df['Engagement_Level'] = df.apply(calculate_engagement_level, axis=1)

# 显示新计算的数据
print(df[['Engagement_Score', 'Engagement_Level']].head())

# 保存更新后的 CSV
df.to_csv("cleaned3_Viral_Social_Media_Trends.csv", index=False)

# 可视化 Engagement Level 分布
plt.figure(figsize=(8, 6))
df['Engagement_Level'].value_counts().sort_index().plot(
    kind='bar', color=['#32CD32', '#FFD700', '#FF6347']  # Low-Medium-High
)
plt.title("Engagement Level Distribution")
plt.xlabel("Engagement Level")
plt.ylabel("Number of Videos")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("处理完成，已保存为 cleaned3_Viral_Social_Media_Trends.csv")
