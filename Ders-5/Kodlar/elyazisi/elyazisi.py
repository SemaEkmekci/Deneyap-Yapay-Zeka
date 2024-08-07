''' MNIST (Modified National Institue of Standarts) veri seti içerisinde 
yer alan 60000 adet 28x28 piksel boyutuna sahip farklı insanların el yazmalarından kesilmiş 
resimlerden oluşmaktadır. Bunlara karşılık gelen 60000 tane sayı (skaler) vardır. MNIST veri setini 
yapay sinir ağları kullanarak veri seti içerisinde yer alan resmin hangi rakama ait olduğunu 
tahminleyen bir model oluşturunuz (Veri seti keras kütüphanesi içerisinden otomatik 
çekilmektedir).
''' 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Flatten 
from keras.models import Sequential 
from keras.utils import to_categorical 
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape) 
print(X_test.shape) 
temp = []

for i in range(len(y_train)): 
 temp.append(to_categorical(y_train[i], num_classes=10)) 
y_train = np.array(temp) 

# Convert y_test into one-hot format 
temp = [] 

for i in range(len(y_test)): 
 temp.append(to_categorical(y_test[i], num_classes=10)) 
y_test = np.array(temp) 

model = Sequential() 
model.add(Flatten(input_shape=(28,28))) 
model.add(Dense(16, activation='sigmoid')) 
model.add(Dense(16, activation='sigmoid')) 
model.add(Dense(16, activation='sigmoid')) 
model.add(Dense(10, activation='softmax')) 

model.summary() 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(X_train, y_train, epochs=300, validation_data=(X_test,y_test))
model.save('mnist_model.h5')
predictions = model.predict(X_test) 
predictions = np.argmax(predictions, axis=1) 
fig, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20, 4)) 

for i in range(10): 
 axes[i].set_title(predictions[i]) 
 axes[i].imshow(X_test[i], cmap='gray') 
 axes[i].get_xaxis().set_visible(False) 
 axes[i].get_yaxis().set_visible(False) 

plt.show()