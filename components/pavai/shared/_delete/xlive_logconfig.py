import logging
import logging.config
#from pythonjsonlogger import jsonlogger
import re
# pip install python-json-logger

## Keep sensitive information out of logs

class SensitiveDataFilter(logging.Filter):
    pattern = re.compile(r"\d{4}-\d{4}-\d{4}-\d{4}")

    def filter(self, record):
        # Modify the log record to mask sensitive data
        record.msg = self.mask_sensitive_data(record.msg)
        return True

    def mask_sensitive_data(self, message):
        # For example, redact credit card numbers like this
        message = self.pattern.sub("[REDACTED]", message)
        return message

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "sensitive_data_filter": {
            "()": SensitiveDataFilter,
        }
    },
    "formatters": {
        "json": {
            "format": "%(asctime)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%SZ",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json",
            "filters": ["sensitive_data_filter"],
        }
    },
    "loggers": {"": {"handlers": ["stdout"], "level": "INFO"}},
}

logging.config.dictConfig(LOGGING)    

# LOGGING = {
#     "loggers": {"": {"handlers": ["stdout"], "level": "ERROR"}},
# }

#logging.handlers.RotatingFileHandler('live_log.log', maxBytes=1000000, backupCount=5)

# # Create a logger
# def get_logger(logname="live_log",level=logging.DEBUG):
#     logger = logging.getLogger(logname)
#     logger.setLevel(level)
#     handler = logging.handlers.RotatingFileHandler(logname+'.log', maxBytes=1000000, backupCount=5)
#     logger.addHandler(handler)
#     return logger

## Keep sensitive information out of logs
## Avoid logging sensitive data
## Mask or redact sensitive data
## ---------------------------------------
# credit_card_number = "1234-5678-9012-3456"
# logger.info(f"User made a payment with credit card number: {credit_card_number}")

