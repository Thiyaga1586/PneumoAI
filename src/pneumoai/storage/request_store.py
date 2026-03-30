from pathlib import Path

from pneumoai.common.settings import settings


def save_request_image(request_id: str, raw_bytes: bytes, filename: str) -> str:
    runtime_dir = Path(settings.runtime_dir) / "requests" / request_id
    runtime_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name if filename else "upload.bin"
    file_path = runtime_dir / safe_name
    file_path.write_bytes(raw_bytes)

    return str(file_path)