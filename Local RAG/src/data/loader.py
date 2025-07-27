import os
import requests
import logging

def download_pdf(url: str, save_path: str):
    try:
        if not os.path.exists(save_path):
            response = requests.get(url)
            if response.ok:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                logging.info(f"Downloaded to {save_path}")
            else:
                raise ValueError(f"Failed to download PDF. Status code: {response.status_code}")
        else:
            logging.info(f"File already exists: {save_path}")
    except Exception as e:
        logging.error(f"Error downloading PDF: {e}")
        raise
