import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests
from ddgs import DDGS
from PIL import Image
from requests.exceptions import RequestException

from trail_lens.classes import CLASS_TO_SEARCH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_extension_dict = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def generate_queries(query: str) -> list[str]:
    """
    Generate search query variations for a species.
    This should help with training.
    """

    QUERY_SUFFIXES = [
        "leaves",
        "plant",
        "close up",
        "in forest",
        "identification",
    ]

    return [f"{query} {s}" for s in QUERY_SUFFIXES]


def search(query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Search DuckDuckGo Images using generated query variations.
    """
    results = []
    for q in generate_queries(query):
        logger.info(f"Searching {q}, limit {max_results}...")
        q_results = DDGS().images(q, max_results=max_results)
        for r in q_results:
            r["search_query_custom"] = q
        results.extend(q_results)
        logger.info(f"Found {len(results)} for {q}")
    return results


def normalize_query_for_storage(query: str) -> str:
    """
    Normalize a query into a filesystem-safe snake_case name.
    1. lowercase
    2. replace whitespace with "_"
    3. remove non-alphanumeric chars
    4. collapse duplicate underscores maybe
    5. trim leading/trailing underscores
    """
    re_separators = r"[\s-]+"
    re_invalid_chars = r"[^a-z0-9_]+"
    re_duplicate_underscores = r"_+"

    return re.sub(
        re_duplicate_underscores,
        "_",
        re.sub(
            re_invalid_chars,
            "",
            re.sub(re_separators, "_", query.lower().strip()),
        ),
    ).strip("_")


def normalize_content_header(content_type: str) -> str:
    """Normalize an HTTP content-type header value."""
    return content_type.split(";")[0].strip().lower()


def get_image_extension(content_type: str) -> str | None:
    """Map an HTTP content-type header to a file extension."""
    return image_extension_dict.get(normalize_content_header(content_type))


def is_valid_image(image_path: Path) -> bool:
    """Validate an image file using Pillow verification."""
    try:
        with Image.open(image_path) as im:
            im.verify()
            return True
    except Exception:  # not sure what verify raises
        return False


def safe_delete_image(image_path: Path) -> None:
    """Delete an image file while suppressing cleanup errors."""
    try:
        if image_path is not None and image_path.exists():
            image_path.unlink()
    except Exception:
        logger.exception(f"Unable to cleanup failure {image_path}.")


# "title": str,
# "image": str,
# "thumbnail": str,
# "url": str,
# "height": int,
# "width": int,
# "source": str
def download(images: list[dict[str, Any]], query: str) -> None:
    """
    Download, validate, and persist candidate images and metadata.
    """
    logger.info(f"Downloading images and metadata {len(images)} for {query}...")

    saved = 0
    download_failures = 0
    save_failures = 0
    unsupported_content_types = 0
    invalid_images = 0
    skipped_existing = 0

    headers = {"user-agent": "trail-lens/0.0.1"}

    normalized_query = normalize_query_for_storage(query)
    folder_path = Path("data/raw_candidates") / normalized_query
    folder_path.mkdir(parents=True, exist_ok=True)

    metadata_path = folder_path / "metadata.jsonl"

    existing_urls = set()
    if not metadata_path.exists():
        next_index = 1
    else:
        line_count = 0
        with open(
            metadata_path,
        ) as file:
            for line in file:
                line_count += 1
                try:
                    json_line = json.loads(line)
                    existing_urls.add(json_line["image_url"])
                except Exception:
                    logger.warning("Unable to parse metadata json line.")
        next_index = line_count + 1

    for image in images:
        image_path: None | Path = None
        try:
            logger.info(f"Processing image {image['title']} ...")

            image_url = str(image["image"])

            if image_url in existing_urls:
                skipped_existing += 1
                continue

            existing_urls.add(image_url)

            r = requests.get(image_url, timeout=10, headers=headers)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "")
            image_extension = get_image_extension(content_type)
            if image_extension is None:
                logger.info(f"Unsupported image extension {content_type} for {image_url}")
                unsupported_content_types += 1
                continue

            image_file_name = f"{normalized_query}_{next_index:06}.{image_extension}"
            image_path = folder_path / image_file_name

            with open(image_path, "wb") as file:
                file.write(r.content)

            if not is_valid_image(image_path):
                safe_delete_image(image_path)
                invalid_images += 1
                continue

            with open(metadata_path, "a") as metadata_file:
                metadata = {
                    "query": query,
                    "file": image_file_name,
                    "title": image["title"],
                    "width": image["width"],
                    "height": image["height"],
                    "image_url": image["image"],
                    "source_url": image["url"],
                    "query_custom": image["search_query_custom"],
                }
                json.dump(metadata, metadata_file)
                metadata_file.write("\n")
            next_index += 1
            saved += 1
        except RequestException:
            logger.warning(f"Unable to fetch {image_url}")
            download_failures += 1
        except OSError:
            if image_path is not None and image_path.exists():
                safe_delete_image(image_path)

            logger.exception(f"Unable to save {image_path}")
            save_failures += 1

    total = len(images)
    success_rate = 0 if total == 0 else saved / total * 100
    logger.info(
        f"total: {total}\n"
        f"saved: {saved}\n"
        f"download failures: {download_failures}\n"
        f"save failures: {save_failures}\n"
        f"unsupported content type: {unsupported_content_types}\n"
        f"invalid images: {invalid_images}\n"
        f"skipped existing images: {skipped_existing}\n"
        f"success rate: {success_rate}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Trail-lens training utilities",
        description="Collect images for a specific species or all starter species.",
    )

    parser.add_argument("query", nargs="?", help="species/search query")
    parser.add_argument(
        "--all", action="store_true", help="download images for all starter species"
    )
    parser.add_argument("--max-results", type=int, help="limit the result set", default=50)

    args = parser.parse_args()

    if args.all:
        for species in CLASS_TO_SEARCH:
            download(search(species, args.max_results), species)
    elif args.query:
        download(search(args.query, args.max_results), args.query)
    else:
        parser.error("Provide a query or use --all")
