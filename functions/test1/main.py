# main.py

from aimodel import load_data, recommend_platform


print("welcome to the Social Media Platform Recommendation System!")
    
hashtag = input("please input #Hashtag）: ").strip()
content_type = input("please input content_type（such as video/image/text）: ").strip()
region = input("please input region(such as US/UK/Asia）: ").strip()

# load data
df = load_data()

# recommend platform based on user input
platform, score = recommend_platform(df, hashtag, content_type, region)

if platform:
    print(f"\n recommend platform：{platform}")
    print(f" avg Engagement Score：{score:.2f}")
else:
    print("\n No matching data found for the given criteria.")


