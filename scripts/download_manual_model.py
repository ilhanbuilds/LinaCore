"""Explicitly download a user-selected model into the ignored local model root."""

import argparse

from lina_core.core.paths import ProjectPaths
from lina_core.security.paths import safe_resolve


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual integration modeli indirir.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("İndirme için --allow-download açıkça verilmelidir.")
    model_root = ProjectPaths.discover().model_dir
    output = safe_resolve(model_root, args.output)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=output,
    )
    print(f"Model local ve ignore edilen dizine indirildi: {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
