from argparse import ArgumentParser
from pathlib import Path
import shutil


DATASET = "trainingdatapro/portrait-and-30-photos-test"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES_DIR / "portrait_test_images"


def main() -> None:
    parser = ArgumentParser(
        description="Download the portrait Kaggle dataset and copy images into examples/."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Number of images to copy from the dataset (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Where to save the copied images (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "kagglehub is required. Install it with: pip install kagglehub"
        ) from exc

    dataset_path = Path(kagglehub.dataset_download(DATASET))
    image_paths = sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    selected_images = image_paths[: args.max_images] if args.max_images > 0 else image_paths
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected_images:
        destination = args.output_dir / image_path.name
        shutil.copy2(image_path, destination)
        print(f"Saved {destination}")

    print(f"Copied {len(selected_images)} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
