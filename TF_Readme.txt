Projekt zakłada implementację algorytmów przegladu hiperparametrów dla modeli uczenia maszynowego.
Stworzone zostały dwa algorytmy - Grid Search oraz Random Search.

Najważniejsze cechy tych algorytmów to:

Grid Search:
	> Grid Search bada wszystkie możliwe kombinacje hiperparametrów zdefiniowanych we wstępnie zdefiniowanej siatce.
	> Systematycznie generuje i ocenia modele dla każdej kombinacji hiperparametrów w sieci.
	> Wyszukiwanie w siatce jest deterministyczne i gwarantuje, że każda kombinacja w siatce zostanie oceniona.
	> Może to być kosztowne obliczeniowo, gdy przestrzeń hiperparametrów jest duża lub liczba hiperparametrów jest wysoka.

Random Search:
	>Random Search bada przestrzeń hiperparametrów, losowo próbkując kombinacje hiperparametrów.
	>Losowo wybiera zestaw hiperparametrów ze zdefiniowanej przestrzeni wyszukiwania i ocenia odpowiedni model.
	>Wyszukiwanie losowe nie obejmuje całej przestrzeni wyszukiwania.
	>Zapewnia bardziej elastyczne podejście i może potencjalnie przypadkowo odkryć dobre kombinacje hiperparametrów.
	>Random Search może być bardziej wydajny obliczeniowo niż wyszukiwanie w siatce, zwłaszcza gdy liczba hiperparametrów jest wysoka.

Podsumowując, główną różnicą między Grid Search a Random Search jest strategia, której używają do eksploracji przestrzeni hiperparametrów. 
Grid Search jest wyczerpujący i obejmuje wszystkie kombinacje we wstępnie zdefiniowanej siatce, podczas gdy Random Search losowo próbkuje kombinacje.
Grid Search gwarantuje, że wszystkie kombinacje zostaną sprawdzone, podczas gdy Random Search zapewnia większą elastyczność, 
ale może nie obejmować całej przestrzeni wyszukiwania. 
Wybór między Grid Search a Random Search zależy od konkretnego problemu, wielkości przestrzeni hiperparametru i dostępnych zasobów obliczeniowych.
 


