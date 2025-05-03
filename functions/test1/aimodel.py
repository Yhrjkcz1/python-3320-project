# aimodel.py

import pandas as pd

def load_data(filepath='cleaned2_Viral_Social_Media_Trends.csv'):
    # load data from CSV file
    df = pd.read_csv(filepath)
    df.dropna(subset=['Hashtag', 'Content_Type', 'Region', 'Platform', 'Engagement_Score'], inplace=True)
    return df

def recommend_platform(df, hashtag, content_type, region):
    # filter data based on user input
    filtered = df[
        (df['Hashtag'].str.lower() == hashtag.lower()) &
        (df['Content_Type'].str.lower() == content_type.lower()) &
        (df['Region'].str.lower() == region.lower())
    ]

    if filtered.empty:
        return None, None  # no matching data

    # calculate average Engagement_Score for each platform
    result = filtered.groupby('Platform')['Engagement_Score'].mean().sort_values(ascending=False)

    # get the best platform and its score
    best_platform = result.idxmax()
    best_score = result.max()

    return best_platform, best_score
