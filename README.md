# VI3DR — Visually Impaired 3D Recognition

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-red.svg)](LICENSE)
[![University: Silesian University of Technology](https://img.shields.io/badge/University-Silesian%20University%20of%20Technology-blue.svg)](https://polsl.pl/en/)
![Year: 2026](https://img.shields.io/badge/Year-2026-orange.svg)
[![Project: Master's Thesis](https://img.shields.io/badge/Project-Master%27s%20Thesis-green.svg)](thesis/RAU-MGR-300463-2026.pdf)

VI3DR to prototyp systemu wspomagającego osoby niewidome i słabowidzące w nauce geometrii przestrzennej przez dotykowe poznawanie fizycznych modeli brył. Telefon przesyła obraz do serwera, który wykrywa bryłę i na żądanie analizuje jej widoczną powierzchnię. Informacja zwrotna jest odczytywana przez aplikację mobilną, dzięki czemu użytkownik może nadal trzymać model w dłoniach.

Projekt powstał w ramach pracy magisterskiej **„System przeznaczony dla osób niewidomych do rozpoznawania dotykanych obiektów 3D”**, której autorem jest **Piotr Marcol**, pod kierunkiem **dr. hab. inż. Michała Maćkowskiego, prof. PŚ**, na Politechnice Śląskiej (2026).

Pełny opis rozwiązania, metodyki i wyników znajduje się w [pracy magisterskiej](thesis/RAU-MGR-300463-2026.pdf). Źródła LaTeX rozpoczynają się w [thesis/main.tex](thesis/main.tex).

## Jak działa system

1. Kamera telefonu rejestruje osobę manipulującą bryłą. Aplikacja przesyła klatki JPEG przez WebSocket do komputera w sieci lokalnej.
2. Serwer wykonuje detekcję YOLO, wybiera obiekt o najwyższej pewności i zapisuje wynik wraz z odpowiadającą mu klatką.
3. Po komendzie głosowej **„info”** klient Android wysyła żądanie `info-request`.
4. Serwer wycina wykryty obiekt i porównuje kolory jego pikseli z kolorami powierzchni zapisanymi w bazie wiedzy. Dopasowanie odbywa się w przestrzeni Lab.
5. Komunikat przypisany do rozpoznanej powierzchni wraca jako `info-response:<tekst>` i jest odczytywany przez syntezator mowy telefonu.

Detektor rozpoznaje sześć klas: **sześcian, kula, walec, prostopadłościan, czworościan i stożek**. Analiza powierzchni wymaga dodatkowo odpowiednich wpisów w bazie wiedzy i zgodnej kolorystyki fizycznych modeli. Dołączona [baza wiedzy](server/knowledge_db/knowledge.json) zawiera obecnie powierzchnie sześcianu i komunikaty opisujące ich kolory. Można ją rozszerzać o kolejne bryły i treści edukacyjne.

## Zawartość repozytorium

| Katalog                                | Zawartość                                                                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| [KMP/](KMP/)                           | Klient Kotlin Multiplatform i Compose Multiplatform; pełny scenariusz użytkowy na Androidzie oraz częściowa implementacja iOS.        |
| [server/](server/)                     | Serwer Python: WebSocket, mDNS, detekcja YOLO, analiza powierzchni, baza wiedzy oraz interfejs PySide6 z warstwą HTML/CSS/JavaScript. |
| [training/](training/)                 | Trening, ewaluacja i predykcja modeli Ultralytics YOLO; filtry wejściowe oraz zapisane wyniki eksperymentów i wagi w `runs/`.         |
| [video_to_dataset/](video_to_dataset/) | Ekstrakcja klatek z nagrań wideo do zbioru obrazów.                                                                                   |
| [dataset_splitter/](dataset_splitter/) | Narzędzie do podziału obrazów i adnotacji YOLO na zbiory treningowy i walidacyjny.                                                    |
| [thesis/](thesis/)                     | Praca magisterska w PDF, źródła LaTeX, ilustracje oraz skrypty i zestawienia wyników badań.                                           |

## Uruchomienie systemu

Poniższe polecenia zakładają powłokę bash/zsh i rozpoczęcie pracy w głównym katalogu repozytorium. W Windows należy użyć odpowiednika interpretera z `venv\Scripts\python.exe` i składni zmiennych środowiskowych właściwej dla używanej powłoki.

### Serwer

Wymagane są Python 3.11 i środowisko graficzne dla aplikacji desktopowej. Telefon i komputer powinny znajdować się w tej samej sieci lokalnej.

```bash
cd server
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt

VI3DR_YOLO_MODEL=best.pt \
VI3DR_YOLO_IMAGE_SIZE=1024 \
./venv/bin/python app.py
```

Dołączony `server/best.pt` odpowiada modelowi **YOLO26n RGB 1024 (E3)** z badań. Jawne wskazanie wag jest istotne: domyślna wartość w `settings.py` to ogólny model `yolo11n.pt`, który nie jest modelem wytrenowanym do rozpoznawania sześciu klas brył VI3DR.

Interfejs serwera pokazuje podgląd, wyniki detekcji i stan połączenia. Pozwala również zmienić model `.pt` oraz załadować bazę wiedzy JSON bez ponownego uruchamiania aplikacji.

Alternatywnie można uruchomić backend bez interfejsu PySide6, z podglądem OpenCV:

```bash
VI3DR_YOLO_MODEL=best.pt \
VI3DR_YOLO_IMAGE_SIZE=1024 \
./venv/bin/python server_main.py
```

Najważniejsze ustawienia z [server/settings.py](server/settings.py):

| Ustawienie                | Wartość domyślna / zastosowanie                                           |
| ------------------------- | ------------------------------------------------------------------------- |
| `HOST`, `PORT`            | `0.0.0.0`, `8765`; adres i port nasłuchiwania.                            |
| `VI3DR_YOLO_MODEL`        | `yolo11n.pt`; nazwa modelu lub ścieżka do wag. Dla VI3DR ustaw `best.pt`. |
| `VI3DR_YOLO_IMAGE_SIZE`   | `640`; dla dołączonego modelu E3 ustaw `1024`.                            |
| `VI3DR_YOLO_CONFIDENCE`   | `0.25`; próg pewności detekcji.                                           |
| `VI3DR_YOLO_DEVICE`       | Opcjonalny wybór urządzenia, np. `cpu`, `cuda:0` lub `mps`.               |
| `VI3DR_DETECTION_ENABLED` | `1`; wartość `0` wyłącza detekcję.                                        |

Zmienne `VI3DR_*` są odczytywane ze środowiska przy starcie. Adres i port zmienia się w `settings.py`. Usługa jest ogłaszana przez mDNS jako `VI3DR Server`, typu `_vi3dr._tcp.local.`.

### Aplikacja Android

Wymagane są Android Studio, JDK zgodny z Android Gradle Plugin 8.13.2 (JDK 17), Android SDK 36 oraz urządzenie z Androidem 7.0 / API 24 lub nowszym, kamerą i mikrofonem. Aplikacja korzysta z systemowych usług rozpoznawania i syntezy mowy.

Otwórz katalog `KMP/` w Android Studio lub zbuduj aplikację z głównego katalogu repozytorium:

```bash
cd KMP
./gradlew :composeApp:assembleDebug

# Instalacja na podłączonym urządzeniu z włączonym debugowaniem USB:
./gradlew :composeApp:installDebug
```

Po uruchomieniu aplikacji:

1. Zezwól na dostęp do kamery i mikrofonu.
2. Wyszukaj serwer w sieci lokalnej lub wpisz jego adres IP i port `8765` ręcznie.
3. Ustaw jakość obrazu i limit FPS, a następnie rozpocznij transmisję.
4. Umieść bryłę w kadrze i wypowiedz **„info”**, aby usłyszeć komunikat o widocznej powierzchni.

Przygotowanie stanowiska i sterowanie transmisją należą w obecnym scenariuszu do operatora. Komenda głosowa służy do żądania informacji. Połączenie używa `ws://`; serwer obsługuje jednego klienta jednocześnie.

Kod iOS znajduje się w `KMP/composeApp/src/iosMain/` oraz `KMP/iosApp/`. Obejmuje kamerę i transmisję obrazu, ale nie realizuje pełnego scenariusza dostępnego na Androidzie. Szczegóły budowania poszczególnych platform zawiera [KMP/README.md](KMP/README.md).

## Zbiór danych

Na potrzeby pracy przygotowano **4666 ręcznie adnotowanych obrazów JPEG o rozdzielczości 1920 × 1440**, wyodrębnionych z dziewięciu nagrań. Materiał przedstawia osoby manipulujące sześcioma rodzajami brył i zawiera również klatki tła bez obiektów. Adnotacje wykonano w CVAT i zapisano w formacie YOLO.

| Podzbiór    | Liczba obrazów |
| ----------- | -------------: |
| Treningowy  |           3782 |
| Walidacyjny |            407 |
| Testowy     |            477 |
| **Razem**   |       **4666** |

Podział w badaniach wykonano **według całych nagrań**, tak aby klatki tego samego filmu nie trafiały do różnych podzbiorów. Zbiór testowy pochodził z nagrań wykonanych w innym dniu. Narzędzie `dataset_splitter/` nie zostało użyte do tego podziału; losowy podział podobnych klatek mógłby zawyżać ocenę jakości.

Pełny zbiór obrazów i adnotacji nie jest dołączony do repozytorium. Do własnego treningu należy przygotować zewnętrzny katalog danych z `dataset.yaml`, listami `train.txt`, `val.txt`, `test.txt` oraz odpowiadającymi obrazom plikami etykiet. Przykładowa konfiguracja dla sześciu klas:

```yaml
path: .
train: train.txt
val: val.txt
test: test.txt
nc: 6
names: [cube, sphere, cylinder, cuboid, tetrahedron, cone]
```

Kolejność `names` musi odpowiadać identyfikatorom klas w adnotacjach. Skrypty obsługują względne i bezwzględne ścieżki obrazów; etykiety są wyszukiwane przez zastąpienie elementu ścieżki `images` elementem `labels`.

Opis narzędzi przygotowania danych: [ekstrakcja klatek](video_to_dataset/README.md) i [podział zbioru](dataset_splitter/README.md).

## Trening i ewaluacja

Z głównego katalogu repozytorium:

```bash
cd training
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Dla środowiska NVIDIA CUDA stos PyTorch można zainstalować przed wymaganiami podstawowymi, korzystając z `requirements.cuda.txt`. Zapisany w nim indeks pakietów dotyczy CUDA 13.2 i powinien odpowiadać środowisku obliczeniowemu.

Przykład treningu konfiguracji YOLO26n RGB 1024, z parametrami limitu epok i zatrzymania użytymi dla E3:

```bash
./.venv/bin/python train.py \
  --dataset-dir /sciezka/do/zbioru \
  --model yolo26n.pt \
  --imgsz 1024 \
  --epochs 1000 \
  --patience 50 \
  --name vi3dr-yolo26n-rgb-1024
```

Urządzenie treningowe jest domyślnie wybierane w kolejności CUDA, MPS, CPU; można wskazać je przez `--device`. Parametry modelu podano jawnie, ponieważ domyślna konfiguracja skryptu używa `yolov8n.pt`, rozmiaru `640` i `patience=25`.

Ewaluacja zapisanych wag na podzbiorze testowym:

```bash
./.venv/bin/python test.py \
  --dataset-dir /sciezka/do/zbioru \
  --model runs/vi3dr-yolo26n-rgb-1024/weights/best.pt \
  --imgsz 1024
```

Wariant w odcieniach szarości można przygotować przez dodanie do polecenia treningu `--input-filter filters/grayscale.py`. Trening tworzy wtedy przefiltrowany zbiór oraz dwa warianty wag: `best.pt` dla obrazów już przefiltrowanych i `best_with_filter.pt` dla oryginalnych obrazów RGB. Dostępny filtr Sobela nie był używany w eksperymentach opisanych w pracy.

W [training/runs/](training/runs/) znajdują się wagi, konfiguracje i raporty eksperymentów, m.in. `args.yaml`, `results.csv`, wykresy i wyniki testów. Więcej opcji, w tym wznowienie treningu, obsługę filtrów i predykcję pojedynczych obrazów, opisano w [training/README.md](training/README.md).

## Wyniki badań

W pracy porównano siedem konfiguracji YOLO26, zmieniając skalę modelu (`n`, `s`), rozmiar wejścia (`640`, `1024`) i reprezentację obrazu (RGB, odcienie szarości). Poniższe wartości pochodzą ze **zbioru testowego** (tabela 6.2 pracy); metryki podano w skali 0–1.

| Wariant | Model       | Obraz             |  `imgsz` |     mAP50 |  mAP50–95 |        F1 |
| ------- | ----------- | ----------------- | -------: | --------: | --------: | --------: |
| E1      | YOLO26n     | RGB               |      640 |     0,716 |     0,599 |     0,696 |
| E2      | YOLO26s     | RGB               |      640 |     0,786 |     0,641 |     0,730 |
| **E3**  | **YOLO26n** | **RGB**           | **1024** | **0,862** | **0,719** | **0,809** |
| E4      | YOLO26s     | RGB               |     1024 |     0,844 |     0,729 |     0,787 |
| E5      | YOLO26n     | Odcienie szarości |      640 |     0,488 |     0,360 |     0,526 |
| E6      | YOLO26s     | Odcienie szarości |      640 |     0,560 |     0,449 |     0,615 |
| E7      | YOLO26n     | Odcienie szarości |     1024 |     0,621 |     0,495 |     0,624 |

Za najlepszy kompromis jakości i kosztu obliczeniowego uznano **E3: YOLO26n RGB 1024**. W pomiarach na MacBooku Air z Apple M1 i 8 GB RAM, przy użyciu MPS, czas predykcji wynosił **37,72 ms** (około **26,5 FPS**), wobec **85,26 ms** dla E4. Są to pomiary predykcji modelu, bez pełnego opóźnienia transmisji, analizy powierzchni i odpowiedzi głosowej (tabela 6.6 pracy).

Zwiększenie rozdzielczości poprawiało jakość detekcji, natomiast usunięcie informacji o kolorze obniżało wyniki we wszystkich porównywanych parach. Większy model nie zapewniał przewagi we wszystkich metrykach. Wnioski dotyczą badanego zbioru i konfiguracji opisanych w pracy.

## Status i dalszy rozwój

System jest prototypem badawczym przeznaczonym do pracy w kontrolowanych warunkach. Badania dotyczyły przede wszystkim detektora; użyteczność edukacyjna i działanie z docelowymi użytkownikami wymagają dalszej oceny.

Najważniejsze obszary rozwoju to:

- rozszerzenie i zróżnicowanie zbioru danych oraz sprawdzenie generalizacji na inne osoby, tła, oświetlenie i kolory brył;
- rozbudowa bazy wiedzy i ocena niezawodności rozpoznawania powierzchni;
- wykorzystanie kolejnych klatek do stabilizacji wyników przy zasłonięciu obiektu;
- testy z osobami niewidomymi i słabowidzącymi oraz rozszerzenie sterowania głosowego;
- uzupełnienie funkcjonalności klienta iOS.

## Licencja

Copyright © 2026 Piotr Marcol.

Autorski kod oprogramowania VI3DR oraz dołączone wagi modeli są udostępniane na licencji **GNU Affero General Public License v3.0** (`AGPL-3.0-only`). Pełny tekst znajduje się w pliku [LICENSE](LICENSE). Oprogramowanie jest udostępniane bez gwarancji, na zasadach określonych w licencji.

Projekt wykorzystuje [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), dostępne na AGPL-3.0. Według [zasad licencjonowania Ultralytics](https://www.ultralytics.com/license) licencja ta obejmuje również wytrenowane modele. Zależności i materiały osób trzecich zachowują własne licencje oraz oznaczenia autorstwa.

AGPL pozwala na używanie, modyfikowanie i rozpowszechnianie oprogramowania, także komercyjnie, pod warunkiem spełnienia jej wymagań. Obejmują one udostępnianie odpowiedniego kodu źródłowego przy rozpowszechnianiu objętego nią oprogramowania oraz, zgodnie z sekcją 13, zaoferowanie źródeł zmodyfikowanej wersji użytkownikom korzystającym z niej przez sieć.

Powyższe udzielenie licencji na oprogramowanie nie obejmuje tekstu pracy magisterskiej (PDF i tekstów LaTeX), ilustracji ani logo. Dla tych materiałów nie udzielono w repozytorium odrębnej licencji; zachowane pozostają prawa ich autorów oraz uprawnienia wynikające z przepisów prawa.
