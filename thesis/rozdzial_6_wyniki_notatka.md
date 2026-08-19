# Rozdział 6 — Wyniki badań i ich analiza

## Założenie rozdziału

Rozdział 6 przedstawia **wyniki eksperymentów zdefiniowanych w rozdziale 5 oraz ich interpretację**.

W rozdziale 5 opisano:

- jakie dane wykorzystano;
- jak skonfigurowano trening;
- jakie zmienne badano;
- jakie metryki zastosowano;
- jakie eksperymenty przeprowadzono.

Rozdział 6 odpowiada natomiast na pytania:

> Co uzyskano?

> Jak zmiana badanego parametru wpłynęła na wyniki?

> Która konfiguracja okazała się najkorzystniejsza?

> Jakie błędy popełniają modele?

> Czy uzyskane wyniki są wystarczające z punktu widzenia docelowego systemu?

Nie należy ponownie szczegółowo opisywać konfiguracji treningu ani definicji metryk. W razie potrzeby należy odwołać się do rozdziału 5 oraz części teoretycznej.

---

## Proponowana struktura rozdziału

1. 6.1 Zestawienie wyników treningów
2. 6.2 Wpływ skali modelu na jakość detekcji
3. 6.3 Wpływ rozdzielczości obrazu wejściowego
4. 6.4 Wpływ reprezentacji obrazu na jakość detekcji
5. 6.5 Porównanie najlepszych konfiguracji
6. 6.6 Analiza błędów detekcji
7. 6.7 Ocena działania systemu w scenariuszu użytkowym
8. 6.8 Dyskusja wyników i ograniczeń badań

---

## Uzasadnienie przyjętej struktury

Aktualna struktura rozdziału 6 została dopasowana bezpośrednio do eksperymentów opisanych w sekcji 5.2.3:

- eksperyment A → skala modelu;
- eksperyment B → rozdzielczość;
- eksperyment C → RGB vs grayscale;
- porównanie końcowe → najlepsze konfiguracje.

Osobna sekcja dotycząca rozdzielczości `640 vs 1024` jest potrzebna, ponieważ rozmiar obrazu wejściowego stanowi jedną z głównych zmiennych badanych. Dzięki temu układ rozdziału prowadzi czytelnika wprost:

> metodyka → eksperyment → wynik → interpretacja → wniosek.

### Aktualizacja po `yolo_runs_summary.csv`

Z pliku `yolo_runs_summary.csv` wynika, że finalna macierz eksperymentów obejmuje **7 runów**, a nie pełny układ 2 × 2 × 2.

| Model | RGB 640 | RGB 1024 | grayscale 640 | grayscale 1024 |
|---|---:|---:|---:|---:|
| YOLO26n | ✓ | ✓ | ✓ | ✓ |
| YOLO26s | ✓ | ✓ | ✓ | — |

Brakuje wyłącznie wariantu:

- YOLO26s + grayscale + 1024.

To oznacza, że struktura rozdziału 6 nadal może pozostać oparta na trzech głównych osiach eksperymentalnych:

- skala modelu: YOLO26n vs YOLO26s;
- rozdzielczość: 640 vs 1024;
- reprezentacja obrazu: RGB vs grayscale.

Trzeba jednak pilnować, aby w analizie nie sugerować pełnego układu eksperymentalnego. Wnioski należy formułować dla dostępnych porównań.

### Ustalone fakty, które muszą pozostać spójne w całym rozdziale

- wszystkie 7 runów wykorzystuje ten sam materiał źródłowy, te same adnotacje i ten sam podział na `train`, `val` i `test`;
- warianty grayscale nie stanowią osobnego zbioru danych w sensie merytorycznym — podczas treningu te same obrazy są wcześniej przekształcane do odcieni szarości;
- przetworzone obrazy grayscale są przygotowywane przed treningiem przede wszystkim po to, aby nie wykonywać tej samej konwersji przy każdym odczycie obrazu w kolejnych epokach;
- po zakończeniu treningu wariantu grayscale skrypt tworzy dodatkowy plik wag z warstwą filtrującą na wejściu modelu; finalny model może dzięki temu przyjmować zwykły obraz RGB, a konwersja odbywa się wewnątrz modelu;
- `best.pt` jest wybierany na podstawie wyników walidacyjnych, nie testowych;
- po zakończeniu każdego runu jego `best.pt` jest oceniany na wydzielonym zbiorze testowym;
- wyniki testowe służą do końcowego porównania wcześniej zaplanowanych konfiguracji;
- wszystkie treningi zakończyły się przed osiągnięciem `epochs=1000` w wyniku mechanizmu early stopping;
- ponieważ `close_mosaic=10` odnosił się do ostatnich 10 epok planowanego treningu, a żaden run nie zbliżył się do 1000 epok, augmentacja Mosaic pozostawała aktywna przez cały faktycznie wykonany trening.

W rozdziale 6 nie należy więc pisać, że grayscale korzystał z „innego zbioru danych”. Poprawne określenia to np.:

> wariant danych wejściowych w odcieniach szarości

albo:

> model trenowany z reprezentacją obrazu grayscale.

---


## 6.1 Zestawienie wyników treningów

### Cel sekcji

Przedstawić **ogólny obraz wszystkich przeprowadzonych runów**, zanim rozpocznie się szczegółowa analiza poszczególnych eksperymentów.

Sekcja odpowiada na pytanie:

> Jakie rezultaty osiągnęły wszystkie wytrenowane modele?

Nie należy tutaj jeszcze szczegółowo interpretować każdej różnicy.

### Co powinno się znaleźć

#### Liczba treningów

Wyjaśnić rozróżnienie:

- wykonano łącznie 7 treningów / runów;
- zostały one wykorzystane w kilku eksperymentach porównawczych;
- każdy run zakończył się zapisaniem własnego `best.pt`;
- `best.pt` był wybierany na podstawie wyników walidacyjnych;
- po zakończeniu treningu każdy `best.pt` został oceniony na wydzielonym zbiorze testowym.

To bardzo ważne, żeby jasno pokazać:

```text
train
  ↓
val podczas treningu
  ↓
wybór best.pt
  ↓
koniec treningu
  ↓
ocena best.pt na test
```

Zbiór testowy nie uczestniczy w uczeniu ani wyborze `best.pt`.

#### Główna tabela wyników

Przed tabelą warto krótko doprecyzować sposób rozumienia zbioru danych. Wszystkie modele były trenowane na tym samym materiale źródłowym i tym samym podziale danych. W runach grayscale zmieniała się reprezentacja obrazu podczas uczenia, a nie skład zbioru ani adnotacje.

Możliwe sformułowanie:

> Do przeprowadzenia eksperymentów wytrenowano siedem modeli o architekturze YOLO, wykorzystując ten sam zbiór danych oraz identyczny podział na zbiory treningowy, walidacyjny i testowy. W części eksperymentów obrazy przed treningiem przekształcano do odcieni szarości zgodnie z procedurą opisaną w rozdziale 5.

Najważniejszym elementem tej sekcji powinna być jedna zbiorcza tabela.

Na podstawie `yolo_runs_summary.csv`:

| ID | Model | Dane | imgsz | Epoki | Best epoch | Precision | Recall | mAP50 | mAP50-95 | F1 | GPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R1 | YOLO26n | RGB | 640 | 190 | 140 | 0.90132 | 0.87623 | 0.94480 | 0.87618 | 0.88860 | RTX 4060 |
| R2 | YOLO26s | RGB | 640 | 108 | 60 | 0.86812 | 0.89471 | 0.94341 | 0.88226 | 0.88121 | RTX 4060 |
| R3 | YOLO26n | RGB | 1024 | 211 | 161 | 0.87447 | 0.89102 | 0.95331 | 0.89684 | 0.88267 | L40S |
| R4 | YOLO26s | RGB | 1024 | 114 | 64 | 0.90483 | 0.90184 | 0.95125 | 0.89878 | 0.90333 | L40S |
| R5 | YOLO26n | grayscale | 640 | 203 | 153 | 0.84740 | 0.78525 | 0.86096 | 0.72145 | 0.81514 | A100 |
| R6 | YOLO26s | grayscale | 640 | 280 | 230 | 0.87385 | 0.80283 | 0.86731 | 0.76026 | 0.83684 | A100 |
| R7 | YOLO26n | grayscale | 1024 | 230 | 180 | 0.87348 | 0.81896 | 0.88291 | 0.78410 | 0.84534 | A100 |

Warto jasno napisać w podpisie tabeli, czy przedstawione metryki są:

- wynikami `best.pt` na zbiorze testowym.

Nie mieszać w jednej tabeli wyników val i test.

W tej tabeli metryki powinny zostać opisane jako wyniki `best.pt` na wydzielonym zbiorze testowym.

#### Liczba wykonanych epok

Ponieważ wszystkie treningi zakończyły się przed osiągnięciem maksymalnej liczby 1000 epok w wyniku mechanizmu early stopping, warto podać rzeczywistą liczbę epok każdego runu.

Może to być osobna kolumna tabeli.

Pozwala to również zobaczyć:

- czy poszczególne modele zbiegały w podobnym czasie;
- czy któryś wariant wymagał wyraźnie dłuższego treningu.

Nie należy jednak traktować liczby epok jako bezpośredniej miary jakości modelu.

#### Przebieg uczenia

Nie ma potrzeby pokazywania `results.png` dla wszystkich siedmiu runów.

Byłoby tego za dużo.

Można wybrać:

- jeden reprezentatywny run;
- ewentualnie dwa wyraźnie różniące się warianty.

Wykres powinien pokazywać np.:

- `train/box_loss`;
- `train/cls_loss`;
- `val/box_loss`;
- `val/cls_loss`;
- mAP50;
- mAP50-95.

Celem nie jest analizowanie każdego punktu, ale pokazanie:

- czy trening był stabilny;
- czy metryki się stabilizowały;
- czy występowały oznaki przeuczenia;
- kiedy mniej więcej następował early stopping.

Przy interpretacji krzywych nie należy przypisywać szybszej zbieżności wariantu `s` wyłącznie większej liczbie parametrów, jeżeli nie przeprowadzono eksperymentu pozwalającego potwierdzić taką przyczynę. Można opisać samą obserwację, np. że w danym porównaniu najlepszy wynik walidacyjny wystąpił wcześniej.

Jeżeli po najlepszej epoce metryka walidacyjna ponownie rośnie, nie należy na tej podstawie twierdzić, że dalszy trening na pewno poprawiłby wynik testowy. Można jedynie wskazać, że uzasadniałoby to ponowienie lub kontynuację treningu w celu sprawdzenia tej hipotezy.

#### Czego tutaj nie robić

Nie analizować jeszcze szczegółowo:

- n vs s;
- 640 vs 1024;
- RGB vs grayscale.

Do tego służą kolejne sekcje.

6.1 ma być punktem odniesienia dla całej dalszej analizy.

---

## 6.2 Wpływ skali modelu na jakość detekcji

### Cel sekcji

Przedstawić wyniki eksperymentu A.

Sekcja odpowiada na pytanie:

> Czy zwiększenie skali modelu z YOLO26n do YOLO26s daje praktycznie istotną poprawę jakości detekcji?

Nie używać określenia „wpływ architektury”, ponieważ modele n i s należą do tej samej architektury i różnią się przede wszystkim skalą, liczbą parametrów oraz kosztem obliczeniowym.

### Porównywane warianty

Porównywać tylko takie pary, w których pozostałe warunki były takie same.

Na przykład:

```text
YOLO26n | RGB | 1024
        vs
YOLO26s | RGB | 1024
```

oraz, jeżeli występuje:

```text
YOLO26n | RGB | 640
        vs
YOLO26s | RGB | 640
```

Na podstawie aktualnej macierzy runów dostępne są następujące porównania:

- RGB 640: YOLO26n vs YOLO26s;
- RGB 1024: YOLO26n vs YOLO26s;
- grayscale 640: YOLO26n vs YOLO26s;
- grayscale 1024: brak porównania, ponieważ nie wykonano runu YOLO26s + grayscale + 1024.

Nie należy porównywać bezpośrednio:

```text
YOLO26n 640
vs
YOLO26s 1024
```

jako dowodu wpływu skali modelu, bo równocześnie zmieniają się dwie zmienne.

### Metryki

Porównać:

- mAP50-95 — główne kryterium;
- mAP50;
- precision;
- recall;
- F1.

Jeżeli posiadamy wiarygodne dane:

- czas inferencji;
- rozmiar modelu;
- liczba parametrów;
- ewentualnie zapotrzebowanie na pamięć.

### Forma prezentacji

Dobrze sprawdzi się niewielka tabela. Na podstawie `yolo_runs_summary.csv`:

| Warunki | YOLO26n mAP50-95 | YOLO26s mAP50-95 | Różnica s−n | YOLO26n F1 | YOLO26s F1 | Różnica F1 s−n |
|---|---:|---:|---:|---:|---:|---:|
| RGB 640 | 0.87618 | 0.88226 | +0.00608 | 0.88860 | 0.88121 | -0.00739 |
| RGB 1024 | 0.89684 | 0.89878 | +0.00194 | 0.88267 | 0.90333 | +0.02067 |
| grayscale 640 | 0.72145 | 0.76026 | +0.03881 | 0.81514 | 0.83684 | +0.02169 |
| grayscale 1024 | 0.78410 | — | — | 0.84534 | — | — |

Jeżeli porównujemy dwie rozdzielczości, można zrobić dwie pary w jednej tabeli.

Szczególnie ważna obserwacja do późniejszego opisu: najlepsze testowe mAP50-95 uzyskał YOLO26s RGB 1024 (`0.89878`), ale YOLO26n RGB 1024 jest bardzo blisko (`0.89684`). Różnica wynosi `0.00194`, czyli około `0.19` punktu procentowego.

### Co analizować

Nie wystarczy napisać:

> YOLO26s uzyskał większe mAP50-95.

Trzeba odpowiedzieć:

- o ile wzrosło mAP50-95;
- czy podobnie zmieniły się precision i recall;
- czy różnica występuje również dla mAP50;
- czy większy model poprawił lokalizację obiektów;
- czy poprawa jest duża z punktu widzenia zwiększonej złożoności modelu.

Kluczowym tematem jest kompromis:

```text
jakość detekcji
        vs
koszt obliczeniowy
```

### Potencjalny wniosek

Nie wpisywać z góry.

Po uzyskaniu wyników wniosek może wyglądać np.:

> Zwiększenie skali modelu poprawiło jakość detekcji, jednak przyrost mAP50-95 był stosunkowo niewielki względem zwiększenia kosztu obliczeniowego.

albo przeciwnie:

> Wariant s osiągnął na tyle wyższą skuteczność, że zwiększony koszt jego działania można uznać za uzasadniony.

Wniosek musi wynikać z faktycznych liczb.

---

## 6.3 Wpływ rozdzielczości obrazu wejściowego

### Cel sekcji

Przedstawić eksperyment B.

Sekcja odpowiada na pytanie:

> Czy zwiększenie rozdzielczości wejściowej z 640 do 1024 poprawia skuteczność detekcji brył?

### Dlaczego to ważne

Oryginalne obrazy mają rozdzielczość 1920 × 1440.

Zmniejszenie ich do rozmiaru wejściowego modelu powoduje utratę części informacji.

Przy niższym `imgsz`:

- drobne cechy mogą zostać utracone;
- krawędzie mogą zostać gorzej odwzorowane;
- częściowo zasłonięty obiekt może zawierać mniej użytecznych informacji.

Jednocześnie wyższa rozdzielczość zwiększa:

- koszt obliczeniowy;
- czas inferencji;
- wymagania dotyczące pamięci.

### Porównania

Najlepiej porównać osobno:

- YOLO26n 640 vs YOLO26n 1024;
- YOLO26s 640 vs YOLO26s 1024.

W finalnej macierzy runów dostępne są:

- RGB YOLO26n: 640 vs 1024;
- RGB YOLO26s: 640 vs 1024;
- grayscale YOLO26n: 640 vs 1024;
- grayscale YOLO26s: brak porównania, ponieważ nie wykonano runu YOLO26s + grayscale + 1024.

### Metryki

Tak jak wcześniej:

- mAP50-95;
- mAP50;
- precision;
- recall;
- F1.

Jeżeli mamy wiarygodne pomiary:

- czas inferencji;
- wykorzystanie pamięci.

Tutaj pomiar czasu jest szczególnie interesujący, ponieważ zwiększenie `imgsz` bezpośrednio wpływa na koszt obliczeń.

### Możliwy wykres

Dobrze sprawdzi się prosty wykres:

```text
mAP50-95
   ↑
   |
   |       ● 1024
   |
   |  ● 640
   +----------------
```

dla n i s.

Można również wykorzystać wykres słupkowy pokazujący cztery konfiguracje:

- n/640;
- n/1024;
- s/640;
- s/1024.

Nie przesadzać z liczbą wykresów, jeśli te same dane są już czytelne w tabeli.

Tabela pomocnicza na podstawie `yolo_runs_summary.csv`:

| Model | Dane | mAP50-95 640 | mAP50-95 1024 | Różnica 1024−640 | F1 640 | F1 1024 | Różnica F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | RGB | 0.87618 | 0.89684 | +0.02066 | 0.88860 | 0.88267 | -0.00593 |
| YOLO26s | RGB | 0.88226 | 0.89878 | +0.01652 | 0.88121 | 0.90333 | +0.02212 |
| YOLO26n | grayscale | 0.72145 | 0.78410 | +0.06265 | 0.81514 | 0.84534 | +0.03020 |
| YOLO26s | grayscale | 0.76026 | — | — | 0.83684 | — | — |

Widać, że wzrost rozdzielczości z 640 do 1024 poprawił mAP50-95 we wszystkich dostępnych parach. Największy przyrost wystąpił dla YOLO26n w wariancie grayscale.

### Analiza

Sprawdzić:

- czy wzrost rozdzielczości poprawia oba modele;
- czy poprawa jest podobna dla n i s;
- czy większa różnica występuje w mAP50-95 niż mAP50;
- czy wyższa rozdzielczość poprawia głównie lokalizację ramek;
- czy koszt zwiększenia rozdzielczości jest uzasadniony.

---

## 6.4 Wpływ reprezentacji obrazu na jakość detekcji

### Cel sekcji

Przedstawić eksperyment C:

- RGB vs grayscale.

To prawdopodobnie będzie jeden z najciekawszych fragmentów całej pracy.

Sekcja odpowiada na pytanie:

> Jak usunięcie informacji o barwie wpływa na zdolność modelu do rozpoznawania brył?

### Dlaczego eksperyment jest ważny

Bryły wykorzystane do przygotowania zbioru danych posiadają stałą kolorystykę.

Powstaje więc ryzyko, że model wykorzystuje podczas detekcji nie tylko:

- kształt;
- krawędzie;
- cienie;
- strukturę przestrzenną;

ale również:

- kolor powierzchni.

Wariant grayscale usuwa informację o barwie, ale nadal zachowuje:

- jasność;
- krawędzie;
- cienie;
- teksturę;
- strukturę przestrzenną obrazu.

Dlatego NIE należy pisać, że grayscale sprawdza:

> czy model rozpoznaje obiekty wyłącznie po kształcie.

Poprawniej:

> czy model jest w stanie skutecznie rozpoznawać bryły bez wykorzystania informacji o barwie.

### Porównanie

Porównywane modele powinny mieć:

- tę samą architekturę;
- tę samą skalę;
- ten sam `imgsz`;
- identyczny split;
- identyczne adnotacje;
- możliwie identyczne parametry treningu.

W eksperymencie nie zmienia się skład zbioru danych. Te same obrazy źródłowe są wykorzystywane w obu wariantach, natomiast w runach grayscale przed treningiem przygotowywana jest ich wersja pozbawiona informacji o barwie. Takie przygotowanie danych ogranicza narzut obliczeniowy podczas samego procesu uczenia, ponieważ konwersja nie musi być wykonywana ponownie przy każdym odczycie obrazu.

Po zakończeniu treningu skrypt tworzy dodatkowy plik wag zawierający warstwę wejściową realizującą to samo przekształcenie. Dzięki temu finalny wariant grayscale może podczas inferencji przyjmować bezpośrednio obraz RGB, tak samo jak wariant podstawowy. W analizie należy zatem mówić o **reprezentacji wykorzystanej przez model**, a nie o dwóch niezależnych zbiorach danych.

W finalnej macierzy dostępne są następujące porównania:

- YOLO26n 640: RGB vs grayscale;
- YOLO26s 640: RGB vs grayscale;
- YOLO26n 1024: RGB vs grayscale;
- YOLO26s 1024: brak porównania, ponieważ nie wykonano runu YOLO26s + grayscale + 1024.

### Metryki

Porównać:

- mAP50-95;
- mAP50;
- precision;
- recall;
- F1.

Tutaj szczególnie interesujący może być stosunek:

```text
wynik grayscale / wynik RGB
```

oraz bezwzględny spadek wartości metryki.

Przykładowo:

```text
ΔmAP50-95 = mAP50-95_RGB - mAP50-95_gray
```

Nie ma potrzeby wprowadzania formalnego wzoru, jeżeli wystarczy opis procentowy lub różnica punktów.

Tabela pomocnicza na podstawie `yolo_runs_summary.csv`:

| Model | imgsz | RGB mAP50-95 | grayscale mAP50-95 | Spadek gray względem RGB | RGB F1 | grayscale F1 | Spadek F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n | 640 | 0.87618 | 0.72145 | -0.15473 | 0.88860 | 0.81514 | -0.07346 |
| YOLO26s | 640 | 0.88226 | 0.76026 | -0.12200 | 0.88121 | 0.83684 | -0.04438 |
| YOLO26n | 1024 | 0.89684 | 0.78410 | -0.11274 | 0.88267 | 0.84534 | -0.03733 |
| YOLO26s | 1024 | 0.89878 | — | — | 0.90333 | — | — |

W dostępnych parach grayscale obniża mAP50-95 względem RGB. Jednocześnie w wariancie YOLO26n wzrost rozdzielczości z 640 do 1024 zmniejsza ten spadek.

### Warto pokazać przebieg treningu

Jeżeli RGB i grayscale zachowują się wyraźnie inaczej podczas treningu, tutaj warto pokazać:

- krzywe mAP50-95;
- ewentualnie loss.

Dzięki temu można sprawdzić, czy grayscale:

- uczy się wolniej;
- osiąga plateau wcześniej;
- ma większą różnicę train/val;
- w ogóle nie zbliża się do wyniku RGB.

### Najważniejsza interpretacja

Jeżeli grayscale okaże się wyraźnie słabszy:

nie należy automatycznie stwierdzać:

> model nauczył się rozpoznawać klasy wyłącznie po kolorze.

Można stwierdzić:

> informacja o barwie ma istotny wpływ na skuteczność modelu.

Dopiero dodatkowe eksperymenty z inną kolorystyką brył pozwoliłyby ocenić, czy rzeczywiście występuje silny shortcut learning związany z przypisaniem konkretnych barw do klas.

---

## 6.5 Porównanie najlepszych konfiguracji

### Cel sekcji

Zebrać wyniki wcześniejszych eksperymentów i odpowiedzieć na pytanie:

> Który z badanych wariantów najlepiej spełnia wymagania systemu?

Nie traktowałbym tego jako zupełnie nowego treningu.

To raczej końcowe porównanie rezultatów wcześniejszych eksperymentów.

### Co porównujemy

Wybrać konfiguracje, które w poprzednich sekcjach okazały się najlepsze lub najbardziej interesujące.

Przykładowo:

| Model | Dane | imgsz | Precision | Recall | mAP50 | mAP50-95 | F1 | Uwagi |
|---|---|---:|---:|---:|---:|---:|---:|---|
| YOLO26s | RGB | 1024 | 0.90483 | 0.90184 | 0.95125 | 0.89878 | 0.90333 | najwyższe mAP50-95 i F1 |
| YOLO26n | RGB | 1024 | 0.87447 | 0.89102 | 0.95331 | 0.89684 | 0.88267 | mAP50-95 tylko o 0.00194 niższe od YOLO26s RGB 1024 |
| YOLO26s | RGB | 640 | 0.86812 | 0.89471 | 0.94341 | 0.88226 | 0.88121 | najlepszy wariant 640 pod względem mAP50-95 |
| YOLO26n | RGB | 640 | 0.90132 | 0.87623 | 0.94480 | 0.87618 | 0.88860 | wyższe F1 niż YOLO26s RGB 640 |
| YOLO26n | grayscale | 1024 | 0.87348 | 0.81896 | 0.88291 | 0.78410 | 0.84534 | najlepszy dostępny wariant grayscale dla YOLO26n |

### Kryteria wyboru

Model końcowy nie powinien zostać wybrany wyłącznie na podstawie największego mAP.

W systemie istotne są również:

- koszt obliczeniowy;
- czas inferencji;
- rozmiar modelu;
- potencjalna możliwość działania na słabszym sprzęcie;
- stabilność predykcji.

To jest miejsce na faktyczne rozstrzygnięcie kompromisu:

```text
najwyższa jakość
vs
wystarczająca jakość przy znacznie mniejszym koszcie
```

Na podstawie obecnych wyników szczególnie ważne będzie porównanie YOLO26s RGB 1024 oraz YOLO26n RGB 1024. YOLO26s uzyskał najwyższe mAP50-95, ale przewaga nad YOLO26n wynosi tylko około `0.19` punktu procentowego. Jeżeli YOLO26n okaże się wyraźnie lżejszy lub szybszy w inferencji, może być korzystniejszym wyborem systemowym mimo minimalnie niższego mAP50-95.

### Benchmark kosztu modeli

Do końcowego wyboru konfiguracji można wykorzystać przygotowany benchmark MPS, ale trzeba jasno rozdzielić **koszt samego modelu** od **czasu działania całego systemu**.

Aktualny skrypt `benchmark.py`:

- automatycznie wybiera MPS, jeżeli jest dostępny;
- wykonuje globalny warm-up MPS;
- dla każdego modelu wykonuje 30 predykcji rozgrzewających;
- wykonuje 100 właściwych pomiarów;
- przed rozpoczęciem i po zakończeniu każdej mierzonej predykcji wywołuje synchronizację MPS;
- mierzy pełne wywołanie `model.predict(...)`, a nie wyłącznie czas przejścia przez warstwy sieci;
- zapisuje średnią, medianę, odchylenie standardowe, minimum, maksimum oraz wynikające z czasu wartości FPS;
- oblicza także liczbę parametrów, GFLOPs oraz rozmiar pliku wag.

Jako podstawową wartość do porównania należy wykorzystywać **medianę czasu**, ponieważ pojedyncze pomiary zawierają sporadyczne skoki czasu wykonania związane ze środowiskiem systemowym. Dwie dotychczas wykonane niezależne serie, rozdzielone kilkunastominutową przerwą, dały bardzo podobne mediany; największa różnica pomiędzy odpowiadającymi sobie medianami wynosiła około 3,3%. Wskazuje to na dobrą stabilność kierunku i skali obserwowanych różnic.

#### Ważna poprawka przed finalnym benchmarkiem

Aktualna wersja `benchmark.py` wyszukuje wyłącznie pliki:

```text
*/weights/best.pt
```

Dla zwykłych runów RGB jest to właściwy artefakt. W przypadku wariantów grayscale finalny model używany w systemie zawiera jednak dodatkową warstwę wejściową wykonującą konwersję obrazu. Jeżeli jest ona zapisana jako osobny plik, np. `best_with_filter.pt`, to właśnie ten plik powinien być użyty w benchmarku końcowym.

W przeciwnym razie pomiar dla grayscale opisuje koszt samej wytrenowanej sieci, ale **nie obejmuje kosztu warstwy filtrującej**, która faktycznie będzie działała podczas inferencji w systemie.

Dlatego przed wykorzystaniem tabeli wydajności w pracy należy:

1. wskazać `best.pt` dla wariantów RGB;
2. wskazać finalny model z warstwą wejściową dla wariantów grayscale;
3. uruchomić wszystkie modele w tej samej procedurze benchmarkowej;
4. zachować tę samą liczbę warm-upów i powtórzeń;
5. raportować przede wszystkim medianę czasu.

Trzecia niezależna seria pomiarowa nadal będzie wartościowa, ale ważniejsze od samego zwiększenia liczby serii jest wcześniejsze upewnienie się, że benchmarkowane są **faktyczne artefakty używane podczas wdrożenia**.

#### Co dokładnie oznacza wynik obecnego benchmarku?

Obraz wejściowy tworzony przez skrypt jest syntetyczny i ma od razu rozmiar zgodny z `imgsz` danego modelu. Oznacza to, że benchmark jest dobry do porównania względnego kosztu wariantów `n`, `s`, `640` i `1024`, ale nie powinien być utożsamiany z całkowitym czasem odpowiedzi VI3DR.

Jeżeli w rozdziale 6 ma pojawić się stwierdzenie typu:

> system przetwarza około X klatek na sekundę,

należy wykonać osobny pomiar na rzeczywistych klatkach o rozdzielczości źródłowej 1920 × 1440 i — najlepiej — w pełnym potoku serwerowym. Wtedy pomiar może uwzględniać m.in. przygotowanie wejścia przez Ultralytics oraz pozostały narzut detektora.

### Ważne

Nie pisać o „statystycznej istotności różnic”, jeżeli:

- każdy wariant został wytrenowany tylko raz;
- nie wykonano powtórzeń z różnymi seedami;
- nie wykonano odpowiedniego testu statystycznego.

Można natomiast pisać o:

- różnicy bezwzględnej;
- różnicy procentowej;
- powtarzalnym kierunku zmian pomiędzy porównywanymi konfiguracjami.

---

## 6.6 Analiza błędów detekcji

### Cel sekcji

Nie ograniczać oceny modelu do pojedynczej wartości mAP.

Sekcja odpowiada na pytanie:

> W jakich sytuacjach model popełnia błędy i które klasy są dla niego najtrudniejsze?

### Macierz pomyłek

Podstawowym elementem powinny być macierze pomyłek dla:

- najlepszego modelu;
- ewentualnie dodatkowo modelu grayscale, jeżeli pokazuje ciekawie inny charakter błędów.

Nie ma potrzeby wrzucać macierzy dla wszystkich siedmiu runów.

### Co analizować

#### Pomyłki pomiędzy klasami

Sprawdzić rzeczywiste wyniki.

Potencjalnie interesujące mogą być:

- sześcian ↔ prostopadłościan;
- walec ↔ stożek;
- inne pary widoczne w macierzy.

Nie zakładać wcześniej, że konkretna para jest najczęściej mylona.

#### Background / brak detekcji

Sprawdzić:

- ile obiektów zostało pominiętych;
- dla których klas występuje to najczęściej;
- czy model generuje fałszywe detekcje na obrazach tła.

To szczególnie ważne z punktu widzenia systemu:

```text
false positive
→ użytkownik może otrzymać błędną informację

false negative
→ użytkownik może nie otrzymać informacji
```

### Analiza przykładów obrazów

Bardzo warto pokazać kilka przykładów.

#### Poprawna detekcja trudnego przypadku

Np.:

- znaczne zasłonięcie dłonią;
- nietypowa orientacja.

#### Błędna klasyfikacja

Pokazać:

- obraz;
- klasę rzeczywistą;
- klasę przewidywaną;
- confidence.

#### Brak detekcji

Pokazać przykład, gdzie:

- obiekt jest silnie zasłonięty;
- ma nietypową orientację;
- światło jest niekorzystne.

#### False positive

Jeżeli występują.

### Czego szukać

Powiązań błędów z:

- zasłonięciem obiektu;
- podobieństwem geometrycznym klas;
- orientacją;
- wielkością obiektu;
- położeniem w obrazie;
- światłem;
- tłem;
- kolorystyką;
- osobą występującą na nagraniu.

Nie należy przypisywać przyczyny błędu bez podstaw.

Można pisać:

> Błąd wystąpił w obrazie, w którym większość powierzchni bryły była zasłonięta dłonią, co mogło utrudnić poprawną detekcję.

Nie:

> Model pomylił bryłę, ponieważ była zasłonięta.

---

## 6.7 Ocena działania systemu w scenariuszu użytkowym

### Cel sekcji

Przejść od:

```text
model działa dobrze
```

do:

```text
czy cały system działa wystarczająco dobrze?
```

Jest to ważne, ponieważ celem pracy nie jest wyłącznie wytrenowanie modelu YOLO, ale przygotowanie kompletnego systemu wspomagającego.

### Co można ocenić

#### Poprawność przepływu

Sprawdzić:

- połączenie aplikacji z serwerem;
- przesyłanie obrazu;
- ciągłą detekcję;
- działanie komendy głosowej;
- wysłanie info-request;
- przygotowanie odpowiedzi;
- odebranie jej przez klienta;
- odczytanie za pomocą TTS.

#### Czas odpowiedzi

Jeżeli da się go wiarygodnie zmierzyć, byłaby to bardzo cenna część badań.

Można mierzyć:

- t0 — rozpoznanie żądania;
- t1 — odebranie żądania przez serwer;
- t2 — zakończenie analizy;
- t3 — odebranie odpowiedzi przez aplikację;
- t4 — rozpoczęcie TTS.

Możliwe wartości:

- czas komunikacji;
- czas inferencji;
- czas analizy;
- całkowity czas odpowiedzi.

Nie trzeba mierzyć wszystkich, jeżeli kod tego nie umożliwia.

Najbardziej użyteczna jest:

- długość czasu od wydania żądania do rozpoczęcia przekazywania odpowiedzi.

#### Stabilność transmisji

Można ocenić:

- osiąganą częstotliwość klatek;
- występowanie opóźnień;
- zachowanie podczas szybkiego poruszania bryłą;
- zachowanie podczas analizy uruchomionej w tle;
- czy bieżąca transmisja pozostaje aktywna podczas przygotowania odpowiedzi.

### Test scenariusza

Można zdefiniować prosty scenariusz:

1. uruchomienie serwera;
2. połączenie aplikacji;
3. rozpoczęcie transmisji;
4. użytkownik bierze bryłę;
5. obraca ją w dłoniach;
6. system wykonuje detekcję;
7. użytkownik wydaje komendę;
8. klient wysyła żądanie;
9. serwer przygotowuje odpowiedź;
10. klient odczytuje komunikat.

Następnie opisać:

- które elementy działają;
- jaki jest czas reakcji;
- jakie problemy występują.

### Ważne ograniczenie

Jeżeli testy systemu NIE zostały wykonane z udziałem osób niewidomych, trzeba to jasno napisać.

Nie można wtedy stwierdzić:

> system jest użyteczny dla osób niewidomych.

Można stwierdzić:

> system realizuje technicznie scenariusz przewidziany dla użytkownika niewidomego.

Rzeczywista ocena użyteczności wymagałaby badań z przedstawicielami grupy docelowej.

### Czy ta sekcja musi zostać?

Tak, jeśli do czasu pisania wyników będzie działał cały system.

Jeżeli `SliceAnalyzer` nadal będzie placeholderem, nie możemy udawać pełnego testu scenariusza końcowego.

Wtedy sekcję trzeba ograniczyć do:

- oceny działania zaimplementowanego prototypu

i bardzo wyraźnie zaznaczyć, które etapy nie były jeszcze ukończone.

---

## 6.8 Dyskusja wyników i ograniczeń badań

### Cel sekcji

Spojrzeć na uzyskane wyniki szerzej.

Nie jest to jeszcze końcowe podsumowanie pracy — to rozdział 7.

Tutaj odpowiadamy:

> Na ile wyniki eksperymentów są wiarygodne i jak szeroko można je uogólniać?

### 6.8.1 Ograniczenia zbioru danych

Do omówienia:

#### Liczba obrazów

4666 obrazów jest wystarczające do wykonania eksperymentów, ale zbiór danych jest niewielki w porównaniu z dużymi benchmarkami detekcji.

#### Dane pochodzą z filmów

Obrazy nie są całkowicie niezależnymi obserwacjami.

Kolejne klatki tego samego nagrania są do siebie podobne.

Podział całych filmów pomiędzy splitami ogranicza problem leakage, ale nie usuwa podobieństwa obrazów wewnątrz poszczególnych zbiorów.

#### Nierównowaga klas

Rozkład klas nie jest równomierny.

Przykładowo czworościan ma znacznie więcej obrazów treningowych niż część pozostałych klas.

Trzeba sprawdzić, czy znajduje to odzwierciedlenie w wynikach per-class.

#### Centralne położenie obiektów

Wykres `labels.jpg` pokazuje, że większość bounding boxów znajduje się w centralnym obszarze obrazu.

Jest to zgodne ze scenariuszem użytkowym, ale ogranicza zróżnicowanie położenia obiektów.

#### Rozmiary obiektów

Na podstawie `labels.jpg` można również omówić zakres znormalizowanych szerokości i wysokości bounding boxów.

### 6.8.2 Ograniczenia dotyczące użytkowników i scen

Do uwzględnienia:

- liczba osób uczestniczących w nagraniach;
- część materiałów wykonywana w tych samych warunkach;
- ograniczona liczba pomieszczeń;
- ograniczone zróżnicowanie tła;
- ograniczone zróżnicowanie sprzętu rejestrującego.

Jeżeli wszystkie dane zostały nagrane jednym telefonem:

- nie można jeszcze stwierdzić pełnej odporności na zmianę urządzenia rejestrującego.

### 6.8.3 Kolorystyka brył

To bardzo ważna część dyskusji.

Bryły posiadają stały układ kolorów.

Eksperyment grayscale pokazuje wpływ usunięcia informacji o barwie, ale nie odpowiada całkowicie na pytanie:

> czy RGB model nauczył się związku konkretnego koloru z konkretną klasą?

Fakt, że finalny model grayscale przyjmuje obraz RGB dzięki dołączonej warstwie wejściowej, nie zmienia znaczenia eksperymentu. Warstwa ta usuwa informację o barwie przed przekazaniem danych do części sieci uczonej na reprezentacji grayscale.

Do pełnej weryfikacji potrzebny byłby dodatkowy zbiór:

- tych samych brył pomalowanych inaczej;
- albo kilku zestawów o różnych paletach.

To naturalny kierunek dalszej walidacji.

### 6.8.4 Transfer learning

Jeżeli wykorzystano pretrained weights:

- warto zaznaczyć, że model rozpoczyna trening z reprezentacją cech wyuczoną na innym zbiorze.

Nie jest to wada.

Jest to standardowa metoda transfer learningu.

Jeżeli nie wykonano finalnego eksperymentu from scratch, nie należy wyciągać wniosków o jego wpływie.

### 6.8.5 Pojedyncze treningi

Jeżeli każdy wariant był trenowany tylko raz:

- wyniki mogą zależeć od losowości procesu uczenia;
- `seed=42` zwiększa powtarzalność konkretnego eksperymentu;
- nie daje jednak informacji o wariancji wyników między różnymi seedami.

Dlatego:

- można porównywać uzyskane rezultaty;
- nie należy mówić o istotności statystycznej różnic.

Pełniejsza analiza wymagałaby kilkukrotnego powtórzenia każdego treningu z różnymi wartościami seed.

### 6.8.6 Sprzęt treningowy

Modele były trenowane na różnych kartach GPU zależnie od dostępności.

Ponieważ sprzęt nie stanowił zmiennej eksperymentalnej:

- nie porównujemy jakości modeli ze względu na GPU;
- nie traktujemy czasu całego treningu jako bezpośrednio porównywalnego pomiędzy wszystkimi runami.

Jeżeli porównujemy czas inferencji, powinien być on zmierzony:

- na tym samym urządzeniu i w tych samych warunkach.

Nie używać czasu treningu z A100 i RTX 4060 do porównywania efektywności architektur.

---

## Proponowany przepływ całego rozdziału

```text
6.1 Zestawienie wyników
        ↓
Co w ogóle uzyskaliśmy?

6.2 n vs s
        ↓
Czy większy model pomaga?

6.3 640 vs 1024
        ↓
Czy większa rozdzielczość pomaga?

6.4 RGB vs grayscale
        ↓
Jak ważna jest informacja o kolorze?

6.5 Najlepsze konfiguracje
        ↓
Który wariant wybieramy?

6.6 Analiza błędów
        ↓
Gdzie i dlaczego model zawodzi?

6.7 Cały system
        ↓
Czy model działa poprawnie w prototypie?

6.8 Dyskusja
        ↓
Jak szeroko można interpretować uzyskane wyniki?
```

---

## Elementy graficzne, które warto wykorzystać

### 6.1

- zbiorcza tabela wszystkich runów;
- 1–2 wykresy przebiegu treningu.

### 6.2

- tabela n vs s;
- ewentualnie wykres mAP50-95.

### 6.3

- tabela 640 vs 1024;
- wykres wpływu `imgsz`.

### 6.4

- RGB vs grayscale;
- wykres przebiegu mAP50-95 dla obu wariantów.

### 6.5

- mała tabela najlepszych konfiguracji;
- tabela kosztu modeli: parametry, GFLOPs, rozmiar wag, mediana czasu inferencji i FPS;
- w tabeli wydajności używać finalnych artefaktów wdrożeniowych, w tym modelu z warstwą grayscale dla odpowiednich runów.

### 6.6

- macierz pomyłek najlepszego modelu;
- 3–6 przykładowych predykcji:
    - poprawna;
    - błędna klasa;
    - false negative;
    - ewentualnie false positive.

### 6.7

- ewentualnie diagram / tabela pomiarów czasu całego systemu;
- nie trzeba ponownie przedstawiać architektury systemu.

### 6.8

Nie potrzebuje wielu nowych rysunków.

Można odwołać się do:

- `labels.jpg` z rozdziału 5;
- tabel rozkładu klas;
- wyników wcześniejszych eksperymentów.

---

## Ważne rozróżnienie — walidacja i test

W rozdziale 6 trzeba bardzo konsekwentnie oznaczać źródło każdej wartości.

### Validation

Wyniki val pokazują przebieg treningu i służą do:

- wyboru `best.pt`;
- obserwacji konwergencji;
- early stopping.

Wykresy uczenia mogą więc przedstawiać dane walidacyjne.

### Test

Po zakończeniu każdego runu:

```text
best.pt
   ↓
test
   ↓
końcowe metryki
```

To właśnie wyniki testowe powinny stanowić podstawę głównych tabel porównawczych w rozdziale 6.

Ważne rozróżnienie terminologiczne:

- `test` nie uczestniczył w treningu;
- `test` nie uczestniczył w wyborze najlepszej epoki ani pliku `best.pt`;
- `test` został wykorzystany do końcowej oceny `best.pt` każdego runu i porównania siedmiu wcześniej zaplanowanych konfiguracji;
- jeżeli na podstawie tego porównania zostanie wybrany model wdrożeniowy, nie należy później pisać, że zbiór testowy „nie uczestniczył w żadnym wyborze modelu”. Poprawne jest stwierdzenie, że nie uczestniczył w uczeniu ani wyborze wag w obrębie runu.

W podpisach tabel najlepiej pisać np.:

> Wyniki modeli na wydzielonym zbiorze testowym.

Dzięki temu nie będzie wątpliwości, skąd pochodzą wartości.

---

## Wyniki per-class

Warto sprawdzić, czy `test.py` / Ultralytics zapisuje wyniki dla każdej klasy.

Jeżeli tak, bardzo przydatna może być tabela:

| Klasa            | Precision | Recall | mAP50 | mAP50-95 |
| ---------------- | --------: | -----: | ----: | -------: |
| Sześcian         |       ... |    ... |   ... |      ... |
| Kula             |       ... |    ... |   ... |      ... |
| Walec            |       ... |    ... |   ... |      ... |
| Prostopadłościan |       ... |    ... |   ... |      ... |
| Czworościan      |       ... |    ... |   ... |      ... |
| Stożek           |       ... |    ... |   ... |      ... |

Nie musi być dla każdego runu.

Najbardziej wartościowa będzie dla:

- wybranego końcowego modelu.

Może znaleźć się w 6.5 lub 6.6.

---

## Czas inferencji i GPU memory

Te dane należy traktować ostrożnie.

### Czas treningu

Nie porównywać, jeżeli runy były wykonywane na:

- RTX 4060;
- L40S;
- A100.

Nie byłoby to porównanie modeli w tych samych warunkach.

### Czas inferencji

Benchmark porównawczy jest już przygotowany i był uruchamiany w dwóch niezależnych seriach. Procedura wykorzystuje to samo urządzenie, MPS, 30 inferencji rozgrzewających dla każdego modelu oraz 100 właściwych powtórzeń z synchronizacją MPS przed i po każdym pomiarze.

Do porównań należy używać przede wszystkim mediany czasu. Dotychczasowe dwie serie są zgodne — odpowiadające sobie mediany różniły się maksymalnie o około 3,3%, podczas gdy różnice pomiędzy wariantami `n` i `s` są znacznie większe.

Przed uznaniem benchmarku za finalny trzeba jednak poprawić dobór plików wag dla modeli grayscale. Obecny skrypt wyszukuje `best.pt`, natomiast finalny wariant grayscale używany przez system zawiera dodatkową warstwę wejściową realizującą konwersję obrazu. Końcowe porównanie wydajności powinno więc obejmować faktyczny plik wdrożeniowy z tą warstwą.

Obecny pomiar należy interpretować jako benchmark porównawczy kosztu modeli. Nie jest on pomiarem czasu całego systemu, ponieważ wykorzystuje syntetyczny obraz już dopasowany rozmiarem do `imgsz` i nie obejmuje transmisji, dekodowania obrazu ani pozostałych etapów VI3DR.

### GPU memory

Analogicznie.

Warto wykorzystać tylko wtedy, gdy pomiary pochodzą:

- z tego samego urządzenia;
- z tej samej procedury;
- przy porównywalnej konfiguracji.

Jeżeli mamy wyłącznie wartości z logów treningowych wykonywanych na różnych GPU, lepiej nie traktować ich jako głównego wyniku eksperymentu.

---

## Jak prowadzić analizę wyników

Każda sekcja eksperymentalna powinna mieć mniej więcej ten sam naturalny schemat:

1. Przypomnienie celu

Krótko:

> Celem eksperymentu było...

Nie powtarzać całej metodyki.

2. Prezentacja wyników

Tabela / wykres.

3. Najważniejsze obserwacje

Np.:

- mAP50-95 wzrosło o ...;
- recall spadł;
- precision wzrosło;
- model s potrzebował większych zasobów.

4. Interpretacja

Dlaczego wynik jest istotny dla projektu?

5. Krótki wniosek

1–2 zdania prowadzące do kolejnej części.

Dzięki temu rozdział nie będzie wyglądał jak:

```text
tabela → tabela → tabela → tabela
```

tylko będzie rzeczywistą analizą.

---

## Czego unikać w rozdziale 6

Nie pisać:

> model jest dobry.

Lepiej:

> model osiągnął mAP50-95 równy ..., a najniższą skuteczność odnotowano dla klasy ...

Nie pisać:

> różnica jest znacząca statystycznie.

Jeżeli nie wykonano testów statystycznych.

Nie pisać:

> model rozpoznaje obiekty wyłącznie po kształcie.

Na podstawie grayscale.

Nie pisać:

> grayscale udowadnia, że model nauczył się kolorów.

Można jedynie wskazać wpływ informacji o barwie.

Nie pisać:

> większy model jest lepszy.

Jeżeli ma o 0,5 pp większe mAP, ale wielokrotnie większy koszt.

W tym projekcie istotny jest kompromis.

Nie pisać:

> model podczas treningu dostraja hiperparametry.

Wagi sieci są parametrami modelu; hiperparametry, takie jak `imgsz`, `batch`, `patience` czy ustawienia augmentacji, są konfigurowane przed treningiem.

Nie powtarzać szczegółowo:

- parametrów treningu;
- budowy YOLO26;
- wzorów mAP;
- przygotowania zbioru danych.

To wszystko zostało opisane wcześniej.

---

## Granica pomiędzy rozdziałem 6 i 7

### Rozdział 6

Można napisać:

> Wariant YOLO26n osiągnął wynik tylko nieznacznie niższy od YOLO26s, przy mniejszej liczbie parametrów, dlatego w kontekście przyjętych wymagań stanowi korzystniejszy wariant do wykorzystania w systemie.

To jest wniosek bezpośrednio wynikający z konkretnego eksperymentu.

### Rozdział 7

Dopiero tam:

> Opracowany system spełnił założony cel pracy w zakresie...

oraz:

> Najważniejszym wnioskiem z przeprowadzonych prac jest...

Czyli:

```text
6 → interpretacja konkretnych wyników
7 → odpowiedź na cel całej pracy
```

---

## Kolejność pracy nad rozdziałem 6

Nie zaczynać od pisania akapitów.

Najpierw trzeba zebrać dane.

### Etap 1 — lista finalnych runów

Dla każdego z siedmiu runów zebrać:

- nazwa runu;
- model;
- RGB / grayscale;
- `imgsz`;
- liczba faktycznie wykonanych epok;
- numer najlepszej epoki;
- precision test;
- recall test;
- mAP50 test;
- mAP50-95 test;
- F1 test;
- ścieżka do `best.pt`;
- macierz pomyłek;
- `results.csv`;
- konfiguracja `args.yaml`.

Na tej podstawie powstaje 6.1.

### Etap 2 — eksperyment A: n vs s

Przygotować pary porównawcze:

- n vs s dla RGB 640;
- n vs s dla RGB 1024;
- n vs s dla grayscale 640;
- brak pary dla grayscale 1024, ponieważ nie wykonano wariantu YOLO26s + grayscale + 1024.

Obliczyć:

- różnicę mAP50-95;
- różnicę mAP50;
- różnicę precision;
- różnicę recall;
- różnicę F1.

Na tej podstawie powstaje 6.2.

### Etap 3 — eksperyment B: 640 vs 1024

Dla tego samego modelu:

- RGB YOLO26n: 640 vs 1024;
- RGB YOLO26s: 640 vs 1024;
- grayscale YOLO26n: 640 vs 1024;
- brak porównania grayscale YOLO26s: 640 vs 1024, ponieważ nie wykonano wariantu YOLO26s + grayscale + 1024.

Sprawdzić:

- zmianę mAP50-95;
- zmianę mAP50;
- precision;
- recall;
- F1;
- ewentualny koszt inferencji.

Na tej podstawie powstaje 6.3.

### Etap 4 — eksperyment C: RGB vs grayscale

Przygotować odpowiadające sobie pary:

- ten sam model;
- ten sam `imgsz`;
- RGB vs grayscale.

Dostępne pary:

- YOLO26n 640;
- YOLO26s 640;
- YOLO26n 1024;
- brak pary YOLO26s 1024, ponieważ nie wykonano wariantu YOLO26s + grayscale + 1024.

Sprawdzić:

- mAP50-95;
- mAP50;
- precision;
- recall;
- F1;
- przebieg uczenia;
- confusion matrix.

Na tej podstawie powstaje 6.4.

### Etap 5 — wybór końcowej konfiguracji

Wybrać kilka najlepszych wariantów.

Porównać:

- jakość detekcji na zbiorze testowym;
- wielkość modelu;
- liczbę parametrów;
- GFLOPs;
- medianę czasu inferencji;
- wynikający z niej FPS;
- wymagania sprzętowe.

Przed finalnym zestawieniem wydajności:

- poprawić benchmark tak, aby dla runów grayscale używał finalnego modelu z warstwą wejściową wykonującą konwersję obrazu;
- wykonać kolejną wspólną serię wszystkich siedmiu artefaktów wdrożeniowych;
- opcjonalnie powtórzyć pełną serię jeszcze raz, aby potwierdzić stabilność median;
- nie mieszać benchmarku kosztu modelu z pomiarem całkowitego czasu odpowiedzi systemu.

Na tej podstawie powstaje 6.5.

### Etap 6 — analiza błędów

Dla wybranego modelu zebrać:

- confusion matrix;
- wyniki per-class;
- false positives;
- false negatives;
- błędne klasyfikacje;
- przykłady trudnych obrazów;
- przykłady poprawnych detekcji przy zasłonięciu.

Spróbować powiązać błędy z:

- zasłonięciem;
- orientacją;
- światłem;
- klasą;
- wielkością bounding boxa;
- pozycją na obrazie.

Na tej podstawie powstaje 6.6.

### Etap 7 — test całego systemu

Po ukończeniu `SliceAnalyzer`:

- uruchomić pełny system;
- połączyć aplikację z serwerem;
- wykonać serię żądań;
- sprawdzić ciągłość transmisji;
- sprawdzić poprawność odpowiedzi;
- zmierzyć czas reakcji, jeżeli możliwe;
- odnotować błędy;
- sprawdzić zachowanie przy braku detekcji.

Na tej podstawie powstaje 6.7.

### Etap 8 — ograniczenia

Na końcu zebrać obserwacje dotyczące:

- wielkości zbioru danych;
- nierównowagi klas;
- liczby osób;
- liczby scen;
- jednego urządzenia rejestrującego;
- położenia obiektów głównie w centrum;
- stałej kolorystyki brył;
- pojedynczego treningu każdego wariantu;
- różnych GPU wykorzystanych do treningu;
- braku testów z osobami niewidomymi, jeżeli nie zostaną wykonane;
- ograniczonej liczby klas;
- potencjalnego wpływu tła i oświetlenia.

Na tej podstawie powstaje 6.8.

---

## Najważniejsze dane, które trzeba teraz zebrać

### Już ustalone

- finalna macierz obejmuje 7 runów;
- brakującym wariantem jest YOLO26s + grayscale + 1024;
- wyniki `best.pt` wszystkich runów na zbiorze testowym są zebrane w `yolo_runs_summary.csv`;
- numery najlepszych epok oraz liczba faktycznie wykonanych epok są znane;
- najlepsze mAP50-95 uzyskał YOLO26s RGB 1024 (`0.89878`);
- YOLO26n RGB 1024 uzyskał bardzo zbliżone mAP50-95 (`0.89684`);
- warianty grayscale korzystają z tego samego zbioru i splitu, ale podczas treningu używają wcześniej przygotowanej reprezentacji w odcieniach szarości;
- po treningu grayscale powstaje finalny model z warstwą wejściową wykonującą konwersję z RGB;
- benchmark MPS został wykonany w dwóch niezależnych seriach i wykazuje stabilne mediany;
- do końcowego porównania wydajności należy użyć faktycznych artefaktów wdrożeniowych, a nie zawsze surowego `best.pt`.

### Do zebrania / sprawdzenia

- macierze pomyłek dla wszystkich lub przynajmniej najważniejszych modeli;
- metryki per-class;
- charakter błędów modeli RGB i grayscale;
- przykłady false positive, false negative i błędnej klasyfikacji;
- zależność błędów od zasłonięcia, orientacji, tła i oświetlenia;
- finalny benchmark kosztu modeli po poprawieniu doboru wag grayscale;
- ewentualny osobny benchmark całego potoku na rzeczywistych klatkach 1920 × 1440;
- możliwość wiarygodnego pomiaru całkowitego czasu odpowiedzi systemu;
- kompletność `SliceAnalyzer` przed testem scenariusza użytkowego;
- informacja, czy test scenariusza końcowego będzie techniczny, czy z udziałem osób niewidomych;
- sprawdzenie, czy nierównowaga klas przekłada się na wyniki per-class;
- sprawdzenie, czy centralne położenie i typowe rozmiary bounding boxów widoczne na `labels.jpg` mają związek z przypadkami poprawnej i błędnej detekcji.

---

## Docelowa struktura

```text
6 Wyniki badań i ich analiza

6.1 Zestawienie wyników treningów
    → wszystkie runy i wyniki testowe

6.2 Wpływ skali modelu na jakość detekcji
    → YOLO26n vs YOLO26s

6.3 Wpływ rozdzielczości obrazu wejściowego
    → 640 vs 1024

6.4 Wpływ reprezentacji obrazu na jakość detekcji
    → RGB vs grayscale

6.5 Porównanie najlepszych konfiguracji
    → wybór modelu do systemu

6.6 Analiza błędów detekcji
    → confusion matrix, per-class, przykładowe błędy

6.7 Ocena działania systemu w scenariuszu użytkowym
    → cały klient–serwer–detekcja–odpowiedź

6.8 Dyskusja wyników i ograniczeń badań
    → jak szeroko można uogólnić rezultaty
```

---

## Główna zasada rozdziału

Każda sekcja powinna odpowiadać na konkretne pytanie:

- 6.1 — jakie wyniki uzyskaliśmy?
- 6.2 — czy większy model pomaga?
- 6.3 — czy większa rozdzielczość pomaga?
- 6.4 — jak ważna jest informacja o kolorze?
- 6.5 — który wariant najlepiej nadaje się do systemu?
- 6.6 — gdzie model popełnia błędy?
- 6.7 — czy cały prototyp działa zgodnie z założeniami?
- 6.8 — jakie są granice wiarygodności tych wyników?

Po rozdziale 6 czytelnik powinien już znać nie tylko wszystkie wyniki liczbowe, ale również rozumieć:

> co one oznaczają z punktu widzenia projektowanego systemu.

Rozdział 7 nie powinien już ponownie analizować tabel i wykresów, lecz zebrać najważniejsze wnioski i odpowiedzieć wprost na pytanie, w jakim stopniu udało się zrealizować cel pracy.

---

## Najważniejsza uwaga strukturalna

Jedna rzecz, którą szczególnie warto utrzymać: **6.2, 6.3 i 6.4 powinny odpowiadać dokładnie eksperymentom A–C z metodyki**.

Dzięki temu rozdział 6 nie będzie zbiorem luźnych analiz, tylko bezpośrednią odpowiedzią na plan badań z rozdziału 5.
