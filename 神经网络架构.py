from tensorflow import  keras
from tensorflow.keras import layers
model=keras.Sequential({
    layers.Dense(512,activation="relu"),
    layers.Dense(10,activation="softmax")
})