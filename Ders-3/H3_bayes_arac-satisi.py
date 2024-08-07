import numpy as np
import pandas as pd

# Veriyi okuyup pandas DataFrame'ine dönüştürme
veri = pd.read_csv("car.csv", encoding='unicode_escape')
print(veri)

# Verinin kopyasını oluşturma
veri_copy = veri.copy()
print(veri_copy)
from sklearn import preprocessing

# LabelEncoder nesnesini oluşturma
sayisallastirma = preprocessing.LabelEncoder()

# Kategorik sütunları sayısal değerlere dönüştürme
veri_copy["fiyat"] = sayisallastirma.fit_transform(veri_copy["fiyat"])
veri_copy["onarim"] = sayisallastirma.fit_transform(veri_copy["onarim"])
veri_copy["kapi sayisi"] = sayisallastirma.fit_transform(veri_copy["kapi sayisi"])
veri_copy["kisi sayisi"] = sayisallastirma.fit_transform(veri_copy["kisi sayisi"])
veri_copy["bagaj boyutu"] = sayisallastirma.fit_transform(veri_copy["bagaj boyutu"])
veri_copy["Guvenlik"] = sayisallastirma.fit_transform(veri_copy["Guvenlik"])
veri_copy["satis "] = sayisallastirma.fit_transform(veri_copy["satis "])

print(veri_copy)
# Giriş ve çıkış verilerini ayarlama
girisler = np.array(veri_copy.drop(["satis "], axis=1))
cikis = np.array(veri_copy["satis "])

from sklearn.model_selection import train_test_split

# Veriyi eğitim ve test setlerine böleme
giris_egitim, giris_test, satis_egitim, satis_test = train_test_split(girisler, cikis, test_size=0.3, random_state=109)

from sklearn.naive_bayes import CategoricalNB

# Categorical Naive Bayes modelini oluşturma ve eğitme
model = CategoricalNB()
model.fit(giris_egitim, satis_egitim)
satis_tahmin = model.predict(giris_test)
# print(satis_tahmin)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix oluşturma
cm = confusion_matrix(satis_test, satis_tahmin)
index = ['Kolay', 'Normal', 'Zor', 'Çok kolay'] 
columns = ['Kolay', 'Normal', 'Zor', 'Çok kolay'] 
cm_df = pd.DataFrame(cm, columns, index) 
print(cm_df)                     
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True, fmt="d")
plt.show()

from sklearn import metrics

# Modelin doğruluğunu hesaplama ve yazdırma
print("Modelin Doğruluğu:", metrics.accuracy_score(satis_test, satis_tahmin) * 100)

# Confusion matrix (karmaşıklık matrisi), bir sınıflandırma modelinin performansını değerlendirmek için kullanılan bir araçtır. Bu matris, gerçek sınıf değerleri ile model tarafından tahmin edilen sınıf değerleri arasındaki ilişkiyi gösterir.

ornek = np.array([3, 3, 3, 2, 0, 0]).reshape(1, -1)
tahmin = model.predict(ornek)
print("Tahmin:", tahmin)

