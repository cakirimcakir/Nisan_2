import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veri
df = pd.read_csv("sgdata.csv")

# Kategorik sütunları sayısal yapıyoruz.
le = LabelEncoder()
df['Marital status'] = le.fit_transform(df['Marital status'])
df['Education'] = le.fit_transform(df['Education'])
df['Occupation'] = le.fit_transform(df['Occupation'])

# Income Class hedef değişken oluyor. 100000den fazla geliri olan 1 kalanı 0 oluyor.
df["Income_Class"] = (df["Income"] > 100000).astype(int)

X = df.drop(["Income", "Income_Class", "ID"], axis=1) #x bağımsız değişkenler
y = df["Income_Class"] #y bağımlı değişkenler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Veriyi eğitim ve test olarak ayır

scaler = StandardScaler()  #KNN için veriyi ölçeklendirdik
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5) #model

knn.fit(X_train_scaled, y_train) #eğitim

y_pred_knn = knn.predict(X_test_scaled)#tahminleme

print("KNN - Doğruluk Oranı:", accuracy_score(y_test, y_pred_knn))
print("\nKNN - Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))

#Matris
cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Düşük", "Yüksek"], yticklabels=["Düşük", "Yüksek"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix - KNN")
plt.show()

