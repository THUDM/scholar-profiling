import logging
import datetime
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) #INFO/DEBUG

now_date = datetime.datetime.now()
now_date = now_date.strftime('%Y-%m-%d_%H-%M-%S')

if not os.path.isdir("./log"):
    os.mkdir("./log")

file_handler = logging.FileHandler('./log/' + str(now_date) + '.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)

logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)
logger.addHandler(console_handler)
