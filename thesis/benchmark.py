from pathlib import Path
import gc
import re
import time

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# ============================================================
# KONFIGURACJA
# ============================================================

RUNS_DIR = Path("/Volumes/KINGSTON/vi3dr/VisuallyImpaired3DRecognition/training/runs")

WARMUP = 30
REPEATS = 100

DEFAULT_IMGSZ = 640

OUTPUT_CSV = Path("model_costs_mps.csv")


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")


# ============================================================
# POMOCNICZE
# ============================================================

def synchronize():
    """Czeka na zakończenie operacji wykonywanych przez MPS."""
    if DEVICE == "mps":
        torch.mps.synchronize()


def cleanup():
    """Czyści pamięć pomiędzy modelami."""
    gc.collect()

    if DEVICE == "mps":
        torch.mps.empty_cache()


def get_imgsz_from_name(run_name: str) -> int:
    """
    Wyciąga imgsz z nazwy katalogu.

    Przykład:
    yolo26n_dataset1_cuda_imgsz1024_lr0.01_seed42
                              ^^^^
    -> 1024
    """

    match = re.search(r"imgsz(\d+)", run_name)

    if match:
        return int(match.group(1))

    print(
        f"UWAGA: Nie znaleziono imgsz w nazwie '{run_name}'. "
        f"Używam domyślnie {DEFAULT_IMGSZ}."
    )

    return DEFAULT_IMGSZ


# ============================================================
# GLOBALNY WARM-UP MPS
# ============================================================

def global_mps_warmup():

    if DEVICE != "mps":
        return

    print("\nGlobalny warm-up MPS...")

    dummy = torch.randn(
        1,
        3,
        640,
        640,
        device="mps"
    )

    for _ in range(30):
        _ = dummy * 2.0 + 1.0

    torch.mps.synchronize()

    del dummy
    torch.mps.empty_cache()

    print("Globalny warm-up zakończony.")


# ============================================================
# BENCHMARK POJEDYNCZEGO MODELU
# ============================================================

def benchmark_model(weights_path: Path):

    run_name = weights_path.parents[1].name

    # np. imgsz640 -> 640
    imgsz = get_imgsz_from_name(run_name)

    print("\n" + "=" * 80)
    print(f"Model: {run_name}")
    print(f"Wagi:  {weights_path}")
    print(f"imgsz: {imgsz}")
    print("=" * 80)

    # --------------------------------------------------------
    # Ładowanie modelu
    # --------------------------------------------------------

    model = YOLO(str(weights_path))

    # --------------------------------------------------------
    # Parametry + GFLOPs
    # --------------------------------------------------------

    print("\nInformacje o modelu:")

    layers, params, gradients, gflops = model.info(
        verbose=True,
        imgsz=imgsz
    )

    params_m = params / 1_000_000

    # --------------------------------------------------------
    # Rozmiar wag
    # --------------------------------------------------------

    size_mb = weights_path.stat().st_size / (1024 ** 2)

    # --------------------------------------------------------
    # Obraz testowy
    # --------------------------------------------------------
    #
    # Sztuczny obraz jest wystarczający do porównania
    # kosztu działania modeli.
    #
    # Wszystkie modele dostają wejście zgodne ze swoim imgsz.
    # --------------------------------------------------------

    image = np.zeros(
        (imgsz, imgsz, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # WARM-UP MODELU
    # --------------------------------------------------------

    print(f"\nWarm-up: {WARMUP} inferencji...")

    for _ in range(WARMUP):

        model.predict(
            source=image,
            imgsz=imgsz,
            device=DEVICE,
            verbose=False
        )

    synchronize()

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print(f"Benchmark: {REPEATS} inferencji...")

    times_ms = []

    for i in range(REPEATS):

        synchronize()

        start = time.perf_counter()

        model.predict(
            source=image,
            imgsz=imgsz,
            device=DEVICE,
            verbose=False
        )

        synchronize()

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000

        times_ms.append(elapsed_ms)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{REPEATS}")

    # --------------------------------------------------------
    # STATYSTYKI
    # --------------------------------------------------------

    mean_ms = float(np.mean(times_ms))
    median_ms = float(np.median(times_ms))
    std_ms = float(np.std(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))

    fps_mean = 1000.0 / mean_ms
    fps_median = 1000.0 / median_ms

    print("\nWynik:")
    print(f"  Params:       {params_m:.3f} M")
    print(f"  GFLOPs:       {gflops:.3f}")
    print(f"  Rozmiar:      {size_mb:.2f} MB")
    print(f"  Mean:         {mean_ms:.3f} ms")
    print(f"  Median:       {median_ms:.3f} ms")
    print(f"  Std:          {std_ms:.3f} ms")
    print(f"  Min:          {min_ms:.3f} ms")
    print(f"  Max:          {max_ms:.3f} ms")
    print(f"  FPS mean:     {fps_mean:.2f}")
    print(f"  FPS median:   {fps_median:.2f}")

    result = {
        "model": run_name,
        "imgsz": imgsz,
        "layers": layers,
        "params": params,
        "params_M": params_m,
        "GFLOPs": gflops,
        "size_MB": size_mb,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "FPS_mean": fps_mean,
        "FPS_median": fps_median,
    }

    # --------------------------------------------------------
    # CZYSZCZENIE
    # --------------------------------------------------------

    del model
    del image

    cleanup()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Wyszukiwanie modeli
    # --------------------------------------------------------

    weights = sorted(
        RUNS_DIR.glob("*/weights/best.pt")
    )

    if not weights:
        raise FileNotFoundError(
            f"Nie znaleziono plików:\n"
            f"{RUNS_DIR}/ */weights/best.pt"
        )

    print(f"\nZnaleziono {len(weights)} modeli:")

    for path in weights:
        print(f" - {path}")

    # --------------------------------------------------------
    # Globalny warm-up
    # --------------------------------------------------------

    global_mps_warmup()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    results = []

    for index, weights_path in enumerate(weights, start=1):

        print(
            f"\n\n"
            f"############################################################\n"
            f"# MODEL {index}/{len(weights)}\n"
            f"############################################################"
        )

        try:

            result = benchmark_model(weights_path)

            results.append(result)

        except Exception as e:

            print("\nBŁĄD:")
            print(weights_path)
            print(repr(e))

            cleanup()

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if not results:
        print("\nNie udało się przetestować żadnego modelu.")
        return

    df = pd.DataFrame(results)

    # sortowanie wg nazwy
    df = df.sort_values(
        by="model"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Wyświetlenie
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 100)
    print("WYNIKI KOŃCOWE")
    print("=" * 100)

    columns_to_print = [
        "model",
        "imgsz",
        "params_M",
        "GFLOPs",
        "size_MB",
        "mean_ms",
        "median_ms",
        "std_ms",
        "FPS_mean",
        "FPS_median",
    ]

    print(
        df[columns_to_print].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.4f"
    )

    print(
        f"\nZapisano wyniki do:\n"
        f"{OUTPUT_CSV.resolve()}"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()