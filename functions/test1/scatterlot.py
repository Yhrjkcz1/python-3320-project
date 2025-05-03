import pandas as pd
import matplotlib.pyplot as plt

# 读取已包含 Engagement_Score 和 Engagement_Level 的 CSV 文件
df = pd.read_csv("cleaned2_Viral_Social_Media_Trends.csv")

# 获取分界点
q1 = df['Engagement_Score'].quantile(0.25)
q3 = df['Engagement_Score'].quantile(0.75)

# 设置颜色映射：不同等级用不同颜色
color_map = {
    'Low': '#FF6347',      # 红色
    'Medium': '#FFD700',   # 黄色
    'High': '#32CD32'      # 绿色
}
colors = df['Engagement_Level'].map(color_map)

# 绘制散点图
plt.figure(figsize=(12, 6))
plt.scatter(range(len(df)), df['Engagement_Score'], c=colors, alpha=0.7)

# 添加分界线
plt.axhline(y=q1, color='blue', linestyle='--', label=f'25% Quantile (Low/Medium) = {q1:.4f}')
plt.axhline(y=q3, color='purple', linestyle='--', label=f'75% Quantile (Medium/High) = {q3:.4f}')

# 限制 Y 轴范围
plt.ylim(0, 1.5)

# 设置标题和标签
plt.title("Engagement Score Distribution by Level")
plt.xlabel("Video Index")
plt.ylabel("Engagement Score")
plt.legend()
plt.tight_layout()
plt.show()