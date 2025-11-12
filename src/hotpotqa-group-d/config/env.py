import os

from dotenv import load_dotenv


class Env:
    def __init__(self):
        load_dotenv()
        self.MISTRAL_KEY = os.getenv("MISTRAL_KEY")
