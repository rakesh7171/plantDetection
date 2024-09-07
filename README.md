# plantDetection

RF FILE
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Build and train the RF model
def train_rf_model():
    # Prompt user for dataset path
    csv_path = '/Users/hritik/myenv/apple_disease_dataset4.csv'
    model_path = 'rf_model6.pkl'

    data = pd.read_csv(csv_path)
    X = data[['Water_Level', 'Soil_Moisture', 'Air_Quality', 'Humidity']]
    y = data['Disease_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"RF Model Accuracy: {accuracy:.2f}")
    
    with open(model_path, 'wb') as file:
        pickle.dump(rf_model, file)

# Load the model
def load_rf_model(path='rf_model6.pkl'):
    with open(path, 'rb') as file:
        return pickle.load(file)

# If this script is run directly, train the RF model
if __name__ == '__main__':
    train_rf_model()


# In[ ]:




