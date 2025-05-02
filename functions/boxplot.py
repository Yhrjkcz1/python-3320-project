import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("cleaned2_Viral_Social_Media_Trends.csv")

# 设置图像风格
plt.figure(figsize=(10, 6))

# 绘制箱线图
df.boxplot(
    column='Engagement_Score',
    by='Engagement_Level',
    grid=False,
    patch_artist=True,
    boxprops=dict(facecolor='#ADD8E6'),
    medianprops=dict(color='red'),
)

# 设置图表标题和轴标签
plt.title("Engagement Score Distribution by Level")
plt.suptitle("")  # 去除默认子标题
plt.xlabel("Engagement Level")
plt.ylabel("Engagement Score")
plt.ylim(0, 5)  # 限制Y轴范围，更清晰
plt.tight_layout()
plt.show()
