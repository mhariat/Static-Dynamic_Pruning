import logging
import time
from utils.common_utils import *


class ColorizingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        message = self.format(record)
        stream = self.stream
        stream.write(message)

    def format(self, record):
        message = logging.StreamHandler.format(self, record)
        parts = message.split('\n', 1)
        parts[0] = self.colorize(parts[0], record)
        message = parts[0] + '\n'
        return message

    # color names to indices
    color_map = {
        'black': 0,
        'red': 1,
        'green': 2,
        'yellow': 3,
        'blue': 4,
        'magenta': 5,
        'cyan': 6,
        'white': 7,
    }

    level_map = {
        logging.DEBUG: (None, 'blue', False),
        logging.INFO: (None, 'black', False),
        logging.WARNING: (None, 'yellow', False),
        logging.ERROR: (None, 'red', False),
        logging.CRITICAL: (None, 'green', False),
    }
    csi = '\x1b['
    reset = '\x1b[0m'

    def colorize(self, message, record):
        date, hour, level, message = message.split(' ')
        first_part = '{} {}'.format(date, hour)
        second_part = level
        bg, fg, bold = self.level_map[record.levelno]
        params = []
        if bg in self.color_map:
            params.append(str(self.color_map[bg] + 40))
        if fg in self.color_map:
            params.append(str(self.color_map[fg] + 30))
        if bold:
            params.append('1')
        if params:
            first_part = ''.join((self.csi, ';'.join(params), 'm', first_part, self.reset))
            second_part = ''.join((self.csi, ';'.join(params + ['1']), 'm', second_part, self.reset))
            message = ''.join((self.csi, ';'.join(params), 'm', message, self.reset))
        return ' '.join([first_part, second_part, message])


def get_logger(name, logpath, displaying=False, saving=True):
    create_dir(logpath)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', "%Y-%m-%d %H:%M:%S")
    log_filename = name + time.strftime("-%Y%m%d-%H%M%S")
    if saving:
        info_file_handler = logging.FileHandler(filename=os.path.join(logpath, log_filename))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = ColorizingStreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
