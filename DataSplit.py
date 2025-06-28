# Importuję potrzebne biblioteki
import pandas as pd
from sklearn.model_selection import train_test_split

"""
W tej części dzielę dane na następujące segmenty:

Training set - 60 %
Validation set - 20 %
Test set - 20 %
"""

# Importuję dane
diabetes = pd.read_csv("diabetes.csv")

# Dzielę dane według podanej wcześniej zasady. Najpierw podział na features i labels
features = diabetes.drop("Outcome", axis=1)
labels = diabetes["Outcome"]

# Podział na testowe i walidacyjne.
# Trzeba powtórzyć operację, ponieważ train_test_split() domyślnie nie podzieli nam danych na trzy części.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Upewniam się, że podział został przeprowadzony zgodnie z założeniami.
print(len(labels), len(y_train), len(y_test), len(y_val))

for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset)/len(labels), 2))

# Zapisuję dane do użycia
X_train.to_csv("train_features.csv", index=False)
X_val.to_csv("val_features.csv", index=False)
X_test.to_csv("test_features.csv", index=False)

y_train.to_csv("train_labels.csv", index=False)
y_val.to_csv("val_labels.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)