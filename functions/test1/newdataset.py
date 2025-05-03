import pandas as pd
import matplotlib.pyplot as plt

# 读取原始 CSV 文件
df = pd.read_csv("Viral_Social_Media_Trends.csv")  # 替换为你的文件名或路径

# 先删除不需要的列，比如 Post_ID
df = df.drop(columns=['Post_ID'])

# 定义 Engagement Score 和 Level 的计算函数
def calculate_engagement(views, likes, shares, comments):
    if views == 0:
        return 0, "No Views"
    score = (1 * likes + 3 * shares + 2 * comments) / views
    return score

# 计算 Engagement Score
df['Engagement_Score'] = df.apply(
    lambda row: calculate_engagement(row['Views'], row['Likes'], row['Shares'], row['Comments']),
    axis=1
)

# 重新划分 Engagement_Level，基于 Engagement_Score 的四分位数
q1 = df['Engagement_Score'].quantile(0.25)  # 第25百分位数
q3 = df['Engagement_Score'].quantile(0.75)  # 第75百分位数

# 根据四分位数划分等级
def categorize_level(score):
    if score <= q1:
        return 'Low'
    elif score <= q3:
        return 'Medium'
    else:
        return 'High'

# 应用新等级划分
df['Engagement_Level'] = df['Engagement_Score'].apply(categorize_level)

# 显示更新后的数据
print(df[['Engagement_Score', 'Engagement_Level']].head())

# 保存更新后的文件
df.to_csv("cleaned2_Viral_Social_Media_Trends.csv", index=False)

# 生成图表：各等级的视频数量
plt.figure(figsize=(8, 6))
df['Engagement_Level'].value_counts().plot(kind='bar', color=['#FF6347', '#FFD700', '#32CD32'])
plt.title("Engagement Level Distribution")
plt.xlabel("Engagement Level")
plt.ylabel("Number of Videos")
plt.xticks(rotation=0)
plt.show()

print("处理完成，已保存新文件为 cleaned2_Viral_Social_Media_Trends.csv")
