import logging
import os
from datetime import datetime

LOG_FILE = (
    f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Creates dynamic logfile
)
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)  # Setting up log path
os.makedirs(logs_path, exist_ok=True)  # creating logs folder

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)  # Creates log directory

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# # For testing purposes
# if __name__ == "__main__":
#     logging.info("Logging has started")
