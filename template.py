import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]:%(message)s')

list_of_files = [
    "src/__init__.py",
    "src/ingestion.py",
    "src/retrival.py",
    "src/utils.py",
    "setup.py",
    "requirements.txt",
    "app.py"
]


for file in list_of_files:
    file_path = Path(file)

    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Making directory:{file_dir} for {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,"w") as f:
            pass
        logging.info("Making left directory")
    else:
        logging.info("All file already excist")
        