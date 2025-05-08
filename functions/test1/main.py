from social_media_engagement_model import SocialMediaEngagementModel

if __name__ == "__main__":
    # Initialize model
    model = SocialMediaEngagementModel('cleaned2_Viral_Social_Media_Trends.csv')
    
    # Load data
    model.load_and_process_data()
    
    # Print sample Engagement_Level to verify calculation
    print("Engagement_Level sample:")
    print(model.df[['Likes', 'Shares', 'Comments', 'Views', 'Engagement_Level']].head())
    
    # Check for data issues
    print("Data issues:", model.check_data_issues())
    
    # Run the rest of the pipeline
    model.encode_features()
    model.split_data()
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.plot_training_curves()
    
    # Get and print evaluation results
    accuracy, report = model.evaluate_model()
    print("准确率:", accuracy)
    print("\n分类报告:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"{label}: {metrics:.4f}")