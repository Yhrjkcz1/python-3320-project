from AI_model.EngagementClassifier1 import EngagementClassifier1
from AI_model.EngagementClassifier2 import EngagementClassifier2

classifier1 = EngagementClassifier1()
classifier1.read_dataset()
classifier1.train_all_models()


classifier2 = EngagementClassifier2()
classifier2.read_dataset()
classifier2.train_all_models()