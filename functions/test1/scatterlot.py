import pandas as pd
import matplotlib.pyplot as plt

# 读取包含 Engagement_Score 和 Engagement_Level 的文件
df = pd.read_csv("cleaned3_Viral_Social_Media_Trends.csv")

# 设置颜色映射
color_map = {
    'Low': '#FF6347',      # 红色
    'Medium': '#FFD700',   # 黄色
    'High': '#32CD32'      # 绿色
}
colors = df['Engagement_Level'].map(color_map)

# 设置分类阈值（与业务规则一致）
low_medium_threshold = 0.05
medium_high_threshold = 0.1

# 绘制散点图
plt.figure(figsize=(12, 6))
plt.scatter(range(len(df)), df['Engagement_Score'], c=colors, alpha=0.7, edgecolors='k', linewidths=0.3)

# 添加分界线
plt.axhline(y=low_medium_threshold, color='blue', linestyle='--', label=f'Low/Medium Threshold = {low_medium_threshold:.2f}')
plt.axhline(y=medium_high_threshold, color='purple', linestyle='--', label=f'Medium/High Threshold = {medium_high_threshold:.2f}')

# 设置图表元素
plt.ylim(0, 0.4)
plt.title("Engagement Score Distribution by Level", fontsize=14)
plt.xlabel("Video Index")
plt.ylabel("Engagement Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
