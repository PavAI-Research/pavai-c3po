from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
import pavai.shared.system_types as system_types

class DataSecurityEngine(system_types.Singleton):
    # --------------------------------
    # Analyze and anonymize PHII data
    # -------------------------------
    _default_operators = {"DEFAULT": OperatorConfig("redact", {})}
    _custom_operators = {
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<not available>"}),
        "CREDIT_CARD": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 8, "from_end": True}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"}),
        "PERSON": OperatorConfig("redact", {})
    }

    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._entities = ["PERSON", "CREDIT_CARD",
                          "PHONE_NUMBER", "EMAIL_ADDRESS", "URL", "LOCATION"]

    def check_text_pii(self, input_text: str, language: str = "en", entities: list = None):
        if entities is None:
            entities = self._entities
        pii_results = self._analyzer.analyze(
            text=input_text, entities=entities, language=language)
        if pii_results:
            logger.debug("PII detected")
            logger.debug(pii_results)
        return pii_results

    def anonymize_text(self, input_text: str, analyzer_results: list, operators: list = None):
        if operators is None:
            return self._anonymizer.anonymize(text=input_text, analyzer_results=analyzer_results,
                                              operators=operators).text
        else:
            return self._anonymizer.anonymize(text=input_text, analyzer_results=analyzer_results,).text


    def test_data_security(self):
        datasecurity = DataSecurityEngine()
        input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
        # input_text = "Hello, how are you?"
        pii_results = datasecurity.check_text_pii(input_text)
        deid_text = datasecurity.anonymize_text(input_text=input_text,
                                                analyzer_results=pii_results,
                                                operators=datasecurity._custom_operators)
        print("ORIGINAL TEXT:", input_text, "\n")
        print("Anonymized TEXT:", deid_text, "\n")

# if __name__ == "__main__":
#     test_data_security()
