# Keras kütüphanesinin Sequential fonksiyonu ile boş bir yapay sinir ağı modeli oluşturmak için gerekli olan sınıf çağrılır.
from keras.models import Sequential
#  Keras kütüphanesinin Dense fonksiyonu ile yapay sinir ağı tam bağlı katmanlarını tanımlamak için gerekli olan sınıf çağrılır. 
from keras.layers import Dense
# matematiksel dizi işlemlerini gerçekleştirmek için Numpy kütüphanesi çağrılır.
import numpy as np
# veri setini bölmek için sklearn.model_selection kütüphanesinden train_test_split fonksiyonu çağrılır. 
from sklearn.model_selection import train_test_split 
# model eğitiminden elde edilen sonuçları çizdirmek için matplotlib.pyplot kütüphanesi çağrılır. 
import matplotlib.pyplot as plt
# sklearn kütüphanesinden metrics fonksiyonu ile modelin değerlendirebilmesi için gerekli fonksiyon çağrılır.
from sklearn import metrics

''' numpy kütüphanesinin genfromtxt özelliğini kullanarak veri seti ile aynı konumda 
olan kod dosyasının içerisindeki INTELLIGENT IRRIGATION SYSTEM.csv veri dosyasını “,” 
ayracına göre (delimiter) okuyarak dataset değişkenine aktarır. '''
dataset = np.genfromtxt('INTELLIGENT IRRIGATION SYSTEM (1).csv', delimiter=',')
# dataset değişkenindeki veri setindeki ilk iki sütunu ([1:,0:2] ) giris parametresi olarak ayarlayarak “Giris” değişkenine aktarır
Giris=dataset[1:,0:2]
# dataset değişkenindeki veri setindeki üçüncü sütunu ([1:, 2] ) çıkış parametresi olarak ayarlayarak “Cikis” değişkenine aktarır. 
Cikis =dataset[1:,2]
# veri setindeki eğitim ve test verilerini %80 eğitim, %20 test olacak şekilde “test_size=0,2” komutu kullanarak rastgele ayırır. 
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, 
test_size=0.2, random_state=0)
# “Sequential” fonksiyonu ile yeni bir boş yapay sinir ağı modeli oluşturarak YSA modelinin ismini “model” değişkenine aktarır. 
model = Sequential()
# YSA modeli “add (Dense)” kodu ile 6 nöronlu gizli katman, 2 girişli (“input_dim”) ve Relu aktivasyon fonksiyonu ilk katmanı oluşturur. 
model.add(Dense(6, input_dim=2, activation='relu'))
# 6 nöronlu gizli katman ve Relu aktivasyon fonksiyonlu ikinci katmanı oluşturur. 
model.add(Dense(6, activation='relu'))
# 6 nöronlu gizli katman ve Relu aktivasyon fonksiyonlu üçüncü katmanı oluşturur. 
model.add(Dense(6, activation='relu'))
# 6 nöronlu gizli katman ve Relu aktivasyon fonksiyonlu dördüncü katmanı oluşturur. 
model.add(Dense(6, activation='relu'))
# 1 nöronlu çıkış katmanı ve sigmoid aktivasyon fonksiyonlu son katmanı oluşturur. 
model.add(Dense(1, activation='sigmoid'))
''' kod satırında “compile” fonksiyonu ile kayıp veri değerlerini “binary_crossentropy”,
optimizasyon yöntemi olarak “adam” ve ölçüm metriği olarak “accuracy” parametrelerine göre 
derler. '''

'''
“binary_crossentropy”, ifadesini kayıp fonksiyonu iki sınıflı problemlerde kullanılır.
Örneğin çalışmada kullanılan veri 
setinde pompa motorunun açık veya kapalı olması iki sınıflı bir problemdir. Bu aşamada 
devreye bu kayıp fonksiyonu girmektedir. Örneğin pompa motor çalışıyor ise 1, çalışmıyorsa 
0 olarak işaretleyebilir.
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# yapay sinir ağı modelindeki nöronların, giriş çıkış parametrelerine ait özellikleri “.summary” fonksiyonu ile görür. 
model.summary()
''' “model.fit” komutu yazarak giriş eğitim verilerine göre çıkış eğitim verileri için Yapay 
sinir ağları ile eğitim gerçekleştirir. Burada her bir eğitim için “batch_size” fonksiyonuna aktarılan 5 değeri 
ile eğitime alınacak veri sayısı belirlenerek 30 adet eğitim (epochs=30) gerçekleştirilir. 
'''
model.fit(Giris_train, Cikis_train, epochs=60, batch_size=5) 
# YSA ile eğitim işlemi sonunda giriş test verilerine göre “.predict” fonksiyonu ile pompa motorunun çalışıp çalışmaması durumu için tahminini gerçekleştirir. 
Cikis_pred = model.predict(Giris_test)
''' YSA modeli sınıflandırma işleminde kullanılan “sigmoid” fonksiyonundan dolayı 
(sigmoid fonksiyonu 0 ile 1 arasında lineer olmayan değerler üretir) model elde edilen sonuçlar 0,5’ten 
büyük ise “1” sınıfına, 0,5’den küçük ise “0” sınıfına aktarılarak ikili sınıflandırma yapılmıştır. Elde edilen 
iki boyutlu dizi sonuçları “.flatten” fonksiyonu kullanılarak tek boyutlu diziye çevrilmiştir. 
'''
Cikis_pred=(Cikis_pred>0.5).flatten()
# YSA modelinden elde edilen tahmin sonuçları ile gerçek sonuçlar karşılaştırılarak bulunun doğruluk değeri “print” komutu kullanılarak konsola yazdırılır.
print("Doğruluk:",metrics.accuracy_score(Cikis_test, Cikis_pred)) 

#  Tahmin-Test Sonuçlarını Karşılaştırma 
# ------------------------------------------------
# YSA modeli ile eğitilen pamuk tarlası sulama durumu tahminlemesine ait hata matrisini (confusion matrix) kullanarak modelin başarısını ölçer. 29 numaralı komut satırında, “sklearn.metrics” kütüphanesinden “confusion_matrix” özelliği yüklenir. 
from sklearn.metrics import confusion_matrix
# hata matrisini görüntüleyebilmek için gerekli olan “seaborn” kütüphanesini yükler. 
import seaborn as sns
#  hata matrisindeki grafikleri çizebilmek için matplotlib kütüphanesi içerisinde yer alan pyplot fonksiyonunu yükler. 
import matplotlib.pyplot as plt
#  YSA modelden elde edilen veri seti üzerindeki işlemleri yapabilmek için gerekli olan “pandas” kütüphanesini yükler
import pandas as pd 
# cm değişkeni ile oluşturulan hata matrisinin satır (Cikis_test) ve sütunlarını (Cikis_tahmin) oluşturur.
cm= confusion_matrix(Cikis_test,Cikis_pred)
#  index ve colums değişkenleri kullanarak hata matrisinde yer alacak olan metinleri belirler. 
index = ['Çalışmıyor','Çalışıyor'] 
columns = ['Çalışmıyor','Çalışıyor']
# cm_df değişkeni ile Cikis_test ve Cikis_tahmin değerlerinin veri çerçevesine (DataFrame) aktarılmasını sağlar. 
cm_df = pd.DataFrame(cm,columns,index)
# “plt.figure” komutu ile 10x6 cm çerçeve boyutunda boş bir çizim ekranı açar. 
plt.figure(figsize=(10,6))
# “sns.heatmap komutu ile oluşturulan veri çerçevesini renkli olarak çizer. Burada annot=True ile sayısal değerler gösterilirken, fmt=”d” ile sayısal değerler tam sayı olarak gösterilir
sns.heatmap(cm_df, annot=True,fmt="d")
plt.show()



'''
# Sonuç Değerlendirme
# -------------------------------
Pamuk tarlası sulama durumuna ait toplam 200 adet veri setinde (Web Kaynağı 5.1) 
kayıtlı verinin %80’ini (160) eğitim, %20’i (40) test eğitim seti olarak ayırmıştı. YSA modeli 160 
kayıt ile eğitmişti. Geriye kalan 40 kayıt (Giris_test) ise modeli tahmin etmek için kullandı 
(Cikis_tahmin).  40 test kaydına ait gerçek sonuçlar (Cikis_test) ile tahmin sonuçlarını karşılaştırdı ve hata matrisi elde etti. Böylece, matristeki sayılar toplamının test verisi olan 40’a eşit olduğu görüldü.


Hata matrisi incelendiğinde, 40 adet test kaydında pamuk tarlasında pompa motorun çalışıp 
çalışmama durumuna ait 10 tane çalışmıyor test verisinden 8 adetinin doğru (çalışmıyor), 2 
adetinin ise yanlış (çalışıyor) tahmin edildiği görülmüştür. 30 adet çalışıyor test verisinden 30 
adetin doğru (çalışıyor), 0 adetin yanlış (çalışmıyor) tahminlediği görülmüştür.  YSA modeli ile pamuk tarlası sulama sistemi için konsol ekranında %95 başarım oranında tahminlendi.
'''