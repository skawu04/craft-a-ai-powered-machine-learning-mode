import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocess data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Random Forest classification report: \n", classification_report(y_test, y_pred_rf))

# Train Neural Network model
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
y_pred_nn = nn_model.predict(X_test) > 0.5
print("Neural Network accuracy: ", accuracy_score(y_test, y_pred_nn))
print("Neural Network classification report: \n", classification_report(y_test, y_pred_nn))

# Send notification via email
def send_notification(model, accuracy, report):
    msg = MIMEMultipart()
    msg['From'] = 'your-email@gmail.com'
    msg['To'] = 'recipient-email@gmail.com'
    msg['Subject'] = 'Model Notification'
    body = f"Model: {model}\nAccuracy: {accuracy}\nClassification Report:\n{report}"
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(msg['From'], 'your-password')
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    server.quit()

send_notification('Random Forest', accuracy_score(y_test, y_pred_rf), classification_report(y_test, y_pred_rf))
send_notification('Neural Network', accuracy_score(y_test, y_pred_nn), classification_report(y_test, y_pred_nn))