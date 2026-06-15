from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import yaml

from dataset import (
    ResolvedDatasetConfig,
    read_split_entries,
    read_split_image_paths,
    resolve_image_list_entry,
)
from input_filters import (
    INPUT_FILTER_IDENTITY,
    load_input_filter,
    resolve_filter_source_path,
)


class FilteredDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class FilteredDatasetResult:
    config_path: Path
    root_path: Path
    filter_path: Path
    filter_name: str
    images_written: int
    images_skipped: int
    labels_copied: int


def prepare_filtered_dataset(
    dataset_config: ResolvedDatasetConfig,
    filter_path: str | Path,
    rebuild: bool = False,
    workers: int = 1,
) -> FilteredDatasetResult:
    if dataset_config.filter_path is not None:
        raise FilteredDatasetError(
            f"dataset is already filtered: {dataset_config.config_path}"
        )

    source_filter_path = resolve_filter_source_path(filter_path)
    input_filter, filter_name = load_input_filter(str(source_filter_path))
    if filter_name == INPUT_FILTER_IDENTITY:
        raise FilteredDatasetError("identity filter does not create a derived dataset")

    output_root = dataset_config.root_path / "filters" / _safe_name(filter_name)
    output_root.mkdir(parents=True, exist_ok=True)
    copied_filter_path = copy_filter_source(source_filter_path, output_root, rebuild)
    print(
        f"Preparing filtered dataset '{filter_name}' from {dataset_config.root_path} "
        f"into {output_root}."
    )
    if rebuild:
        print("Existing filtered images will be overwritten.")
    else:
        print("Existing filtered images will be reused; pass --rebuild-filtered-dataset to overwrite.")
    print(f"Filtering workers: {workers}.")

    input_filter.eval()
    split_specs = [
        ("train", dataset_config.train_path),
        ("val", dataset_config.val_path),
    ]
    if dataset_config.test_path is not None:
        split_specs.append(("test", dataset_config.test_path))

    images_written = 0
    images_skipped = 0
    labels_copied = 0
    for split_name, split_path in split_specs:
        split_result = write_filtered_split(
            split_name,
            split_path,
            dataset_config.root_path,
            output_root,
            input_filter,
            rebuild,
            workers,
        )
        images_written += split_result.images_written
        images_skipped += split_result.images_skipped
        labels_copied += split_result.labels_copied

    dataset_yaml_path = write_filtered_dataset_yaml(
        dataset_config,
        output_root,
        copied_filter_path,
    )
    return FilteredDatasetResult(
        config_path=dataset_yaml_path,
        root_path=output_root,
        filter_path=copied_filter_path,
        filter_name=filter_name,
        images_written=images_written,
        images_skipped=images_skipped,
        labels_copied=labels_copied,
    )


@dataclass(frozen=True)
class SplitWriteResult:
    split_name: str
    images_total: int
    images_written: int
    images_skipped: int
    labels_copied: int


@dataclass(frozen=True)
class ImageFilterJob:
    image_path: Path
    output_image_path: Path
    source_label_path: Path
    output_label_path: Path
    should_write_image: bool


@dataclass(frozen=True)
class ImageFilterResult:
    image_written: bool
    image_skipped: bool
    label_copied: bool


def copy_filter_source(source_filter_path: Path, output_root: Path, rebuild: bool) -> Path:
    target_filter_path = output_root / "filter.py"
    if target_filter_path.exists():
        same_content = target_filter_path.read_bytes() == source_filter_path.read_bytes()
        if not same_content and not rebuild:
            raise FilteredDatasetError(
                f"filtered dataset already contains a different filter: {target_filter_path}. "
                "Use --rebuild-filtered-dataset to overwrite it."
            )

    shutil.copy2(source_filter_path, target_filter_path)
    return target_filter_path


def write_filtered_split(
    split_name: str,
    split_path: Path,
    source_root: Path,
    output_root: Path,
    input_filter: torch.nn.Module,
    rebuild: bool,
    workers: int,
) -> SplitWriteResult:
    image_paths = split_image_paths(split_path, source_root)
    print(
        f"Filtered dataset split '{split_name}': {len(image_paths)} image file(s) to process."
    )
    output_entries: list[str] = []
    used_relative_paths: set[Path] = set()
    jobs: list[ImageFilterJob] = []

    for image_path in image_paths:
        image_rel = filtered_image_relative_path(
            image_path,
            source_root,
            split_name,
            used_relative_paths,
        )
        output_image_path = output_root / image_rel
        output_entries.append(image_rel.as_posix())

        source_label_path = label_path_for_image(image_path)
        output_label_path = output_root / label_relative_path_for_image(image_rel)
        jobs.append(
            ImageFilterJob(
                image_path=image_path,
                output_image_path=output_image_path,
                source_label_path=source_label_path,
                output_label_path=output_label_path,
                should_write_image=rebuild or not output_image_path.exists(),
            )
        )

    images_written, images_skipped, labels_copied = process_filter_jobs(
        jobs,
        input_filter,
        workers,
    )

    split_list_path = output_root / f"{split_name}.txt"
    split_list_path.write_text("\n".join(output_entries) + "\n", encoding="utf-8")
    print(
        f"Filtered dataset split '{split_name}' done: "
        f"written={images_written}, skipped={images_skipped}, labels={labels_copied}."
    )
    return SplitWriteResult(
        split_name=split_name,
        images_total=len(image_paths),
        images_written=images_written,
        images_skipped=images_skipped,
        labels_copied=labels_copied,
    )


def process_filter_jobs(
    jobs: list[ImageFilterJob],
    input_filter: torch.nn.Module,
    workers: int,
) -> tuple[int, int, int]:
    if workers < 1:
        raise FilteredDatasetError("--filter-workers must be at least 1")

    if workers == 1 or len(jobs) <= 1:
        results = [run_filter_job(job, input_filter) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_filter_job, job, input_filter) for job in jobs]
            results = [future.result() for future in as_completed(futures)]

    images_written = sum(1 for result in results if result.image_written)
    images_skipped = sum(1 for result in results if result.image_skipped)
    labels_copied = sum(1 for result in results if result.label_copied)
    return images_written, images_skipped, labels_copied


def run_filter_job(
    job: ImageFilterJob,
    input_filter: torch.nn.Module,
) -> ImageFilterResult:
    image_written = False
    image_skipped = False
    if job.should_write_image:
        apply_filter_to_image_file(job.image_path, job.output_image_path, input_filter)
        image_written = True
    else:
        image_skipped = True

    label_copied = False
    if job.source_label_path.exists():
        job.output_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(job.source_label_path, job.output_label_path)
        label_copied = True

    return ImageFilterResult(
        image_written=image_written,
        image_skipped=image_skipped,
        label_copied=label_copied,
    )


def split_image_paths(split_path: Path, source_root: Path) -> list[Path]:
    if split_path.is_dir():
        return read_split_image_paths(split_path, source_root)

    return [
        resolve_image_list_entry(entry, source_root)
        for entry in read_split_entries(split_path)
        if entry.strip()
    ]


def apply_filter_to_image_file(
    image_path: Path,
    output_image_path: Path,
    input_filter: torch.nn.Module,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FilteredDatasetError(f"cannot read image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        filtered = input_filter(tensor)

    if filtered.ndim != 4 or filtered.shape[0] != 1 or filtered.shape[1] != 3:
        raise FilteredDatasetError(
            "input filter must return a BCHW tensor with one image and 3 channels"
        )

    output_rgb = (
        filtered.squeeze(0)
        .detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), output_bgr):
        raise FilteredDatasetError(f"cannot write image: {output_image_path}")


def filtered_image_relative_path(
    image_path: Path,
    source_root: Path,
    split_name: str,
    used_paths: set[Path],
) -> Path:
    try:
        relative_path = image_path.relative_to(source_root)
    except ValueError:
        relative_path = Path("images") / split_name / image_path.name

    if "images" not in relative_path.parts:
        relative_path = Path("images") / split_name / relative_path.name

    candidate = relative_path
    index = 1
    while candidate in used_paths:
        candidate = relative_path.with_name(
            f"{relative_path.stem}_{index}{relative_path.suffix}"
        )
        index += 1

    used_paths.add(candidate)
    return candidate


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")

    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_relative_path_for_image(image_rel: Path) -> Path:
    parts = list(image_rel.parts)
    try:
        images_index = parts.index("images")
    except ValueError:
        return image_rel.with_suffix(".txt")

    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def write_filtered_dataset_yaml(
    dataset_config: ResolvedDatasetConfig,
    output_root: Path,
    copied_filter_path: Path,
) -> Path:
    payload = {
        "path": ".",
        "train": "train.txt",
        "val": "val.txt",
    }
    if dataset_config.test_path is not None:
        payload["test"] = "test.txt"
    payload.update(
        {
            "nc": dataset_config.config["nc"],
            "names": dataset_config.config["names"],
            "filter": copied_filter_path.relative_to(output_root).as_posix(),
        }
    )

    dataset_yaml_path = output_root / "dataset.yaml"
    with dataset_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return dataset_yaml_path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "filter"
