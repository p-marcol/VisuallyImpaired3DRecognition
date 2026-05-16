# Dataset splitter

Skrypt dzieli dataset obrazow i labeli YOLO na `train` oraz `val` z zachowaniem czestotliwosci klas. Obrazy bez pliku labela albo z pustym plikiem labela sa traktowane jako osobna klasa `empty`.

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uzycie

```bash
python split_dataset.py \
  --images-dir /sciezka/do/images \
  --labels-dir /sciezka/do/labels \
  --output-dir /sciezka/do/output \
  --val-size 0.2 \
  --seed 42
```

Domyslnie wynik zostanie zapisany jako:

```text
output/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Jesli chcesz nadpisac poprzednio wygenerowany podzial, dodaj:

```bash
--overwrite
```

Domyslnie skrypt tworzy pusty plik `.txt` w folderze wynikowym dla obrazow, ktore nie mialy pliku labela. Mozesz to wylaczyc:

```bash
--no-empty-label-files
```

Domyslnie pliki sa kopiowane. Jesli chcesz je przeniesc ze zrodla do folderu wynikowego, dodaj:

```bash
--move
```

Uwaga: `train_test_split(..., stratify=y)` wymaga co najmniej 2 obrazow w kazdej klasie, w tym w klasie `empty`.
