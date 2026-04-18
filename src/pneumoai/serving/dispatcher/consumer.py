from typing import Optional

from pneumoai.serving.dispatcher.worker import process_next_task


class LocalPredictionConsumer:
    def consume_once(self) -> Optional[dict]:
        return process_next_task()