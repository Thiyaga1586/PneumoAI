from pneumoai.common.logging import configure_logging
from pneumoai.common.settings import settings
from pneumoai.queue.worker import run_worker_loop
from pneumoai.storage.sqlite import init_db


def main() -> None:
    configure_logging(settings.log_level)
    init_db()
    run_worker_loop()


if __name__ == "__main__":
    main()