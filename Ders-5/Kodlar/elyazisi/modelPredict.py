import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
import tkinter as tk
from tkinter import filedialog


# Modeli yükleme
model = load_model('mnist_model.h5')

# Resmi ön işleme fonksiyonu
def load_and_preprocess_image(image_path):
    # Resmi yükleme
    img = Image.open(image_path).convert('L')  # Gri tonlamaya çevirme
    img = img.resize((28, 28))  # 28x28 piksele yeniden boyutlandırma
    img_array = np.array(img)
    
    # MNIST verisindeki gibi ölçeklendirme
    img_array = 255 - img_array  # MNIST verisindeki gibi beyaz arkaplan için
    img_array = img_array / 255.0  # Normalizasyon
    
    return img_array

# Tkinter ile dosya seçme penceresi açma
def open_file_dialog():
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    return file_path

# Dosya seçme penceresi aç
while True:
    image_path = open_file_dialog()

    if image_path:
        # Seçilen resmi işleyip modele uygun hale getirme
        img_array = load_and_preprocess_image(image_path)
        img_array = np.expand_dims(img_array, axis=0)  # Modelin beklediği şekilde (1, 28, 28) boyutuna getirme
    
        # Tahmin yapma
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions, axis=1)
    
        # Sonucu gösterme
        print(f'Tahmin edilen rakam: {predicted_label[0]}')
    
        # Resmi gösterme
        plt.imshow(Image.open(image_path).convert('L'), cmap='gray')
        plt.title(f'Tahmin: {predicted_label[0]}')
        plt.axis('off')
        plt.show()
    else:
        print("Hiçbir dosya seçilmedi.")
    