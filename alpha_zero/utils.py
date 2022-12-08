import logging


__all__ = ['load_config', 'set_logger', 'AvgManager']


def load_config(config_path):
    config = dict()
    with open(config_path, 'r') as f:
        config_text = f.read()
    exec(config_text, globals(), config)
    return config, config_text


def set_logger(log_file='../sample.log'):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


class AvgManager:
    def __init__(self):
        self.value = 0.0
        self.n = 0

    def __call__(self):
        return self.value / self.n

    def update(self, value):
        self.value += value
        self.n += 1