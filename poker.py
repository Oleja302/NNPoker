import keyboard
import pandas as pd
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
from keras.callbacks import EarlyStopping

#Get train data from csv
poker_train = pd.read_csv(
    "poker-hand-training-true.data",
    names=["suit_of_card1", "rank_of_card1", "suit_of_card2", "rank_of_card2", "suit_of_card3", "rank_of_card3", 
           "suit_of_card4", "rank_of_card4", "suit_of_card5", "rank_of_card5", "poker_hand"])

#Get test data from csv
poker_test = pd.read_csv(
    "poker-hand-testing.data",
    names=["suit_of_card1", "rank_of_card1", "suit_of_card2", "rank_of_card2", "suit_of_card3", "rank_of_card3", 
           "suit_of_card4", "rank_of_card4", "suit_of_card5", "rank_of_card5", "poker_hand"]) 

# Get features and labels from train data 
features = poker_train.copy()
labels = features.pop("poker_hand") 

#Get features and labels from test data
features_test = poker_test.copy()
labels_test = features_test.pop("poker_hand") 

#Create model 
labels_inv_norm_layer = layers.Normalization(invert=True, axis=None)
labels_inv_norm_layer.adapt(labels)

predictors_norm_layer = layers.Normalization(axis=1)
predictors_norm_layer.adapt(features) 

inputs = keras.Input(shape=(10,)) 
normalized_input = predictors_norm_layer(inputs) 
hidden = layers.Dense(256, activation='tanh')(normalized_input)
hidden = layers.Dense(256, activation='tanh')(hidden) 
outputs = layers.Dense(1, activation='tanh')(hidden) 
denormalized_output = labels_inv_norm_layer(outputs)
model = keras.Model(inputs=inputs, outputs=outputs) 

#Characteristic learning
EPOCHS = 500  # epochs
LEARNING_RATE = 0.005 # learning rate
BATCH = 128   # batch size 

#Compiling and learning
early_stopping = EarlyStopping(patience=30) 
model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy']) 

print('\n# Training')
for epoch in range(EPOCHS + 1): 
    res_train = model.fit(features, labels, epochs=1, verbose=0, batch_size=BATCH, validation_split=0.2, callbacks=[early_stopping]) 
    print(f"epoch: {epoch} | MSE: {format(res_train.history['val_loss'][0], '.4f')} | accuracy: {format(res_train.history['val_accuracy'][0] * 100, '.2f')}%") 
    
print('\n# Testing')
model.evaluate(features_test, labels_test, verbose=1) 

print('\Save neural network')
print("ENTER - yes\nESC - no") 
while True:
    if keyboard.read_key() == 'enter': 
        #Save poker model
        print('\n# The model is saved!')
        model.save('pmodel.keras')
        break 
    elif keyboard.read_key() == 'esc':
        break 

# print('Load neural network')
# print("ENTER - yes\nESC - no") 
# while True:
#     if keyboard.read_key() == 'enter': 
#         #Load poker model 
#         print('\n# The model is loaded') 
#         model = keras.models.load_model('pmodel.keras') 

#         data = poker_test.sample(n = 10) 
#         print(data['poker_hand']) 
#         data.drop(columns=['poker_hand'], inplace=True, axis=1) 

#         prediction = model.predict(data) 
#         print(prediction) 
#         break 
#     elif keyboard.read_key() == 'esc':
#         break 