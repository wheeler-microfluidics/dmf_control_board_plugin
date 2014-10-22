from datetime import datetime
import logging

from path_helpers import path
from pip_helpers import install


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(str(datetime.now()))
    logging.info(install(['-U', '-r',
                          path(__file__).parent.joinpath('requirements.txt')],
                         verbose=True))
