import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

df=pd.read_csv("C:\\Users\Firdevs\Downloads\mnist_train.csv")
print(df.head())
dff=pd.read_csv("C:\\Users\Firdevs\Downloads\mnist_test.csv")
print(dff.head())

# Veri setinin boyutunu görüntüle
print("Train Dataset boyutu :" , df.shape)
print("Test Dataset boyutu :" , dff.shape)

# Train Datasetinde ki sınıf etiketlerinin dağılımını görüntüle
print("Train Dataset sınıf etiketlerinin dağılımı :")
print(df["label"].value_counts())

# Test Datasetinde ki sınıf etiketlerinin dağılımını görüntüle
print("Test Dataset sınıf etiketleri dağılımı:")
print(dff["label"].value_counts())

#Train Datasetinde ki sınıf etiketlerinin dağılımını görselleştir.
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="label")
plt.title("Train Dataseti Sınıf Dağılımı")
plt.show()

#Test Datasetinde ki sınıf etiketlerinin dağılımını görselleştir.
plt.figure(figsize=(10,6))
sns.countplot(data=dff, x="label")
plt.title("Test Dataseti Sınıf Dağılımı")
plt.show()

#Veri setindeki dengesizlik için çapraz doğrulama ( cross -validation) yöntemi kullanıcaz.Kfold çapraz doğrulama

#önce özellikler ve etiketler arası ayırma yapalım.
X=df.drop(columns=["label"])
y=df["label"]

# StandardScaler objesini oluşturun
scaler = StandardScaler()

# Train ve Test veri setlerini birlikte ölçeklendirin
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(dff.drop(columns=["label"]))

# Örnek bir sınıflandırma modeli olan Logistik Regresyon
model = LogisticRegression(max_iter=1000 , class_weight='balanced')

# Çapraz doğrulama için KFold kullanarak bir doğrulama bölme nesnesi oluşturun
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Çapraz doğrulama skorlarını hesaplayın
cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="accuracy")

#Çapraz doğrulama skorlarının ortalamasını ve standart sapmasını yazdıralım.
print("Çapraz doğrulama skorları :", cv_scores)
print("Ortalama skor :", cv_scores.mean())
print("Standart Sapma :", cv_scores.std())