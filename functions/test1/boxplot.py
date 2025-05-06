import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("cleaned3_Viral_Social_Media_Trends.csv")

# 使用 IQR 去除异常值（离群值）
Q1 = df['Engagement_Score'].quantile(0.25)
Q3 = df['Engagement_Score'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

# 过滤掉大于 upper_limit 的值
df_filtered = df[df['Engagement_Score'] <= upper_limit]

# 绘制箱线图（已去除过大的值）
plt.figure(figsize=(6, 6))
plt.boxplot(
    df_filtered['Engagement_Score'],
    patch_artist=True,
    boxprops=dict(facecolor='#87CEFA', color='black'),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none')
)

# 添加标签
plt.title("Engagement Score (Outliers Removed by IQR)", fontsize=14)
plt.ylabel("Engagement Score", fontsize=12)
plt.ylim(0, upper_limit * 1.1)  # 设置 Y 轴范围稍高于上限
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

#compare diff platforms

# 读取数据
df = pd.read_csv("cleaned3_Viral_Social_Media_Trends.csv")

# 去除 Engagement_Score 的离群值（基于每个平台单独处理）
def remove_outliers_iqr(group):
    Q1 = group['Engagement_Score'].quantile(0.25)
    Q3 = group['Engagement_Score'].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    return group[group['Engagement_Score'] <= upper_limit]

# 分组去除离群值
df_filtered = df.groupby('Platform', group_keys=False).apply(remove_outliers_iqr)

# 获取平台顺序
platforms = df_filtered['Platform'].unique()
platforms.sort()

# 准备绘图数据：按平台分组的 Engagement_Score 列表
data = [df_filtered[df_filtered['Platform'] == platform]['Engagement_Score'] for platform in platforms]

# 绘制箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(data,
            patch_artist=True,
            labels=platforms,
            boxprops=dict(facecolor='#ADD8E6', color='black'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))

# 图表标签
plt.title("Engagement Score Distribution by Platform (Outliers Removed)", fontsize=14)
plt.xlabel("Platform")
plt.ylabel("Engagement Score")
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
