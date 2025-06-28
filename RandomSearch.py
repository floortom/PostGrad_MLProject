# Importuję potrzebne biblioteki
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from itertools import product
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""
Tworzę funkcję do random search, przyjmującą model (z założenia z sklearn),
słownik hiperparametrów, liczbę iteracji oraz dane treningowe i walidacyjne.
"""
def random_search(model_name, parameter_space, num_iterations, X_train, y_train, X_val, y_val, X_test, y_test):
    # Zmienna dla najlepszego zestawu hiperparametrów
    best_params = None
    # Zmienna dla odpowiadającego powyższym hiperparametrom wyniku
    best_score = float('-inf')
    # Zmienna dla zestawów kombinacji parametrów i wyniku
    results = []

    # Iteruję podaną ilość
    for _ in range(num_iterations):
        # container na hiperparametry i ich wartości
        params = {}

        # Dla każdej iteracji, randomowo wybieram wartości dla każego hiperparametru
        # oraz przypisuję je do słownika
        for param, values in parameter_space.items():
            params[param] = random.choice(values)

        # Tworzę instancję modelu, podając zestaw hiperparametrów
        # "**" umożliwia podanie hiperparametrów jako kwargsów
        model = model_name(**params)

        # trenuję model na danych treningowych.
        # "ravel()" został użyty do dopasowania formatu danych (column vector -> 1D array)
        model.fit(X_train, y_train.values.ravel())

        # Ewaluacja modelu na danych ewaluacyjnych
        score = model.score(X_val, y_val.values.ravel())

        # Dodanie zestawu wyniku i użytych hiperparametrów do listy
        results.append((score, params))

        # Jeśli obecny wynik jest lepszy od dotychczasowego najlepszego, nastepuje aktualizacja
        if score > best_score:
            best_score = score
            best_params = params

    # Sortuję listę "results" w kolejności malejącej według wartości wyniku.
    # "key=lambda x: x[0]" określa, że wynik (pierwszy element każdego tupla) powinien zostać użyty do sortowania
    results.sort(reverse=True, key=lambda x: x[0])

    # Ustawienie umozliwiające wyprintowanie wyników w przypadku liczby mniejszej od założonych 50 kombinacji
    num_sets_to_print = min(50, len(results))

    # Printowanie efektów
    for i, (score, params) in enumerate(results[:num_sets_to_print], 1):
        print(f"Rank {i}: Score: {score}, Parameters: {params}")

    # Wybór najlepszych 3 wyników do testu
    best_3 = results[:3]

    # Testowanie modelu, zasada działania jw
    for i, (score, params) in enumerate(best_3, 1):
        model_test = model_name(**params)
        model_test.fit(X_test, y_test.values.ravel())
        test_score = model_test.score(X_test, y_test.values.ravel())
        print("------------------------------------------------------------------")
        print(f"For set ranked {i}: Test Score: {test_score}, Parameters: {params}")

    # zwrot najlepszych 3 wyników
    return best_3


"""
Poniżej importuję dane do testu (podział w innym pliku), oraz przygotowuję zestawy hiperparametrów do sprawdzenia
czy grid search działa na różnych modelach i zestawach parametrów.
"""
tr_features = pd.read_csv("train_features.csv")
tr_labels = pd.read_csv("train_labels.csv")
val_features = pd.read_csv("val_features.csv")
val_labels = pd.read_csv("val_labels.csv")
te_features = pd.read_csv("test_features.csv")
te_labels = pd.read_csv("test_labels.csv")

rfc_parameters = {
    "bootstrap": [True, False],
    "max_depth": [2, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"],
    "n_estimators": [5, 10, 50],
    }

rfr_parameters = {
    "bootstrap":         [True, False],
    "max_depth":         [2, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["auto", "sqrt"],
    "n_estimators":      [5, 10, 50],
}

lr_parameters = {
    "penalty": ["l2", None],
    "dual": [False],
    "tol": [0.0001],
    "C": [1.0],
    "fit_intercept": [True, False],
    "intercept_scaling": [1],
    "class_weight": ["balanced", None],
    "random_state": [None],
    "solver": ["newton-cg", "newton-cholesky"],
    "max_iter": [100, 120, 140, 160, 180, 200],
    "multi_class": ["auto", "ovr"],
    "verbose": [0],
    "warm_start": [True, False],
    "n_jobs": [None],
    "l1_ratio": [None],
}

mlpc_parameters = {
    "hidden_layer_sizes": [20, 40, 60, 80, 100],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "warm_start": [True, False],
}

# Użycie random search
random_search(RandomForestClassifier,
                rfc_parameters,
                100,
                tr_features,
                tr_labels,
                val_features,
                val_labels,
                te_features,
                te_labels
                )

