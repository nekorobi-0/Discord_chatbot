"""Download the text embedding weights using a build-only dotenv secret."""

import shutil
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import hf_hub_download


def main() -> None:
    token = dotenv_values("/run/secrets/hf_env", interpolate=False).get("HF_TOKEN", "")
    if not token or not token.strip():
        raise SystemExit("Set HF_TOKEN in .env before building the embedding image.")
    source = hf_hub_download(
        repo_id="ggml-org/embeddinggemma-2-GGUF",
        filename="embeddinggemma-2-BF16.gguf",
        revision="58faf90940886df2d62be555c8765d00dac362c5",
        token=token.strip(),
    )
    destination = Path("/models/embeddinggemma-2.gguf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


if __name__ == "__main__":
    main()
