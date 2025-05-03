import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("cleaned2_Viral_Social_Media_Trends.csv")

# 设置图像大小和样式
plt.figure(figsize=(6, 6))

# 绘制一个总体的 Engagement_Score 箱线图
plt.boxplot(df['Engagement_Score'], patch_artist=True,
            boxprops=dict(facecolor='#87CEFA'),
            medianprops=dict(color='red'))

# 图表标签
plt.title("Overall Engagement Score Distribution")
plt.ylabel("Engagement Score")
plt.ylim(0, 1.5)  # 限制 Y 轴范围

plt.tight_layout()
plt.show()
