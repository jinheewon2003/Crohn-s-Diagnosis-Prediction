import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np


xlsx_file = 'data.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(xlsx_file)

# Filter out antibiotic usage to remove extraneous data
df_filtered = df[df['AB_exposure'] != 'YesAB']

# Change data from strings to numbers
df_filtered = pd.get_dummies(df_filtered, columns=['Diagnosis'], prefix='AsInt')
df_filtered = pd.get_dummies(df_filtered, columns=['race'], prefix='AsInt')
df_filtered = pd.get_dummies(df_filtered, columns=['sample_location'], prefix='AsInt')

# Drop sample, subject, Diagnosis, AB_exposure,
df_filtered = df_filtered.drop(columns=['sample', 'subject', 'AB_exposure', 'AsInt_Not IBD'])

# Add in if PCDAI is NA
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(df_filtered)
df_filtered = pd.DataFrame(imputed_data, columns=df_filtered.columns)

# Define inputs and outputs
X = df_filtered.drop(columns=['AsInt_CD'])
y = df_filtered['AsInt_CD']

# Normalize all of the numbers from 0 to 1 using MaxScalar
scaler = MinMaxScaler()
for column in X.columns:
    X[column] = scaler.fit_transform(X[[column]])

# Initialize split
def random_split(X, y, random_state = 42):
    # random_state is for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = random_split(X, y)

# Initialize model
# model = tf.keras.Sequential([
#     # TODO
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# Initialize model details
# def weighted_loss(y_true, y_pred):
#     # Calculate class weights based on the proportion of 0s and 1s in y_true
#     class_weights = tf.math.reduce_sum(y_true, axis=0) / tf.math.reduce_sum(y_true)
    
#     # Compute weighted loss
#     loss = tf.reduce_mean(tf.multiply(class_weights, tf.square(y_true - y_pred)))
    
#     return loss

# model.compile(optimizer='adam', loss=weighted_loss, loss_weights=[1.0, 0.5])

# Train the model
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
# loss = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)

# # Predict using the trained model
# y_pred = model.predict(X_test)

# # Create threshhold on sigmoid
# threshold = 0.5
# y_pred_binary = (y_pred > threshold).astype(int)

# # Concatenate y_test and y_pred_binary side by side
# y_comparison = np.concatenate((y_test.values.reshape(-1, 1), y_pred_binary), axis=1)

# # Print the comparison
# print("Actual (y_test) vs. Predicted (y_pred_binary):")
# print(y_comparison)

# # MSE Loss
# mse_loss = mean_squared_error(y_test, y_pred_binary)
# print("Mean Squared Error (MSE) Loss:", mse_loss)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

y_pred = model.predict(X_test)

# Print predicted and true values side by side
print("Predicted True")
for pred, true in zip(y_pred, y_test):
    print(f"{pred:<11} {true}")

# Print accuracy
print("Test Accuracy:", accuracy)