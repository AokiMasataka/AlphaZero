import logging


__LOGER_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'waening': logging.WARNING,
    'error': logging.ERROR
}


def set_logger(log_file: str = '../sample.log', level: str = 'info') -> None:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    
    log_level = __LOGER_LEVELS[level]
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


class AvgManager:
    def __init__(self, accuracy: int = 4) -> None:
        self._value: float = 0.0
        self._n: int = 0
        self.accuracy: int = accuracy
    
    def __call__(self) -> float:
        return self._value / self._n
    
    def __len__(self) -> int:
        return self._n
    
    def __add__(self, other: float):
        self._value += other
        self._n += 1
        return self
    
    def __repr__(self) -> str:
        return str(self())