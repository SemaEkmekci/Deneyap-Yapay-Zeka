import numpy as np
import pandas as pd
veri=pd.read_csv("hayvanatbahcesi.csv",encoding='unicode_escape')
girisler=np.array(veri.drop(["hayvan adi","sinifi"],axis=1))
cikis=np.array(veri["sinifi"])

from sklearn.model_selection import train_test_split

# Verileri eğitim-test olarak ayır. Test verisi oranı %35 olsun.
X_train, X_test, y_train, y_test = train_test_split(girisler,cikis,test_size=0.35,random_state=109)

from sklearn.naive_bayes import CategoricalNB

# Naive Bayesian sınıflandırması için model oluştur.
# Model ismi: gnb (Gaussian Naive Bayes)
gnb = CategoricalNB()

# Modeli eğit. X_train verisini modele verince Y_train verisini çıktı olarak vermeyi öğrensin.
gnb.fit(X_train, y_train)
# Model eğitimi tamamlandı, artık veri gönderince tahmin yapabilir.

# X_test verilerini, eğitilmiş (fitted) modele  verince, modelin buna göre sınıf tahminlerini y_pred dizisine al.
y_pred = gnb.predict(X_test) # y_pred: X_test verilerine göre sınıfı tahmin edilen Y değerleri (prediction).

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Normalde sistem %100 doğru çalışsaydı; X_test verisini verince, cevap olarak Y_test verisini dönmesi gerekirdi.
# Bizim eğittiğimiz modele X_test verisini verdiğimizde, y_pred değerini almıştık.
# Şimdi y_pred ile aslında olması gereken Y_test verilerini karşılaştırıyoruz.
# Bunun sonucunda bir hata matrisi çıkıyor karşımıza (cm)
# Hata matrisinde false-positive, false-negative, true-positive ve true-negative miktarlarını görüyoruz.
cm= confusion_matrix(y_test,y_pred)
index = ['1','2','3','4','5','6','7']
columns = ['1','2','3','4','5','6','7']
cm_df = pd.DataFrame(cm,columns,index)
plt.figure(figsize=(10,6))
sns.heatmap(cm_df, annot=True,fmt="d")
plt.show()

from sklearn import metrics

# Son olarak eğittimiğiz modelin, yaptığı tahminlere göre % cinsinden doğruluk değerini elde ediyoruz.
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Veri setinden 1 örnek alarak modele gönderelim, sınıf tahmin sonucunu yazdıralım:
ornek=np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]).reshape(1,-1)
print(gnb.predict(ornek))

# Bir başka örnek:
ornek=np.array([1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]).reshape(1,-1)
print(gnb.predict(ornek))