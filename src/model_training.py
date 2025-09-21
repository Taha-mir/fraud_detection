# src/model_training.py
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
 model = RandomForestClassifier(
 n_estimators=100,
 random_state=42,
 class_weight='balanced',
 n_jobs=-1
 )
 model.fit(X_train, y_train)
 return model