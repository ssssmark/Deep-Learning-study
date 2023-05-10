from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# 神经网络架构
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 设置编译步骤参数（优化器、损失函数、指标）

model.compile(optimizer="rmsprop", # 优化器
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 准备图像数据

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype("float32")/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype("uint8")/255

# 训练模型
model.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss,test_acc=model.evaluate(test_images,test_labels)
print(f"test_acc:{test_acc}")

digit=train_images[4]
plt.imshow(digit,cmap=plt.cm.binary)
print(train_labels[4])


