import logging
import warnings 
from rich.logging import RichHandler
from rich import print,pretty
pretty.install()

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
warnings.filterwarnings("ignore")

