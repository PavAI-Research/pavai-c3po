from src.shared.solar.llmdatasecurity import DataSecurityEngine

def test_data_security():
    datasecurity = DataSecurityEngine()
    input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
    # input_text = "Hello, how are you?"
    pii_results = datasecurity.check_text_pii(input_text)
    deid_text = datasecurity.anonymize_text(input_text=input_text,
                                            analyzer_results=pii_results,
                                            operators=datasecurity._custom_operators)
    print("ORIGINAL TEXT:", input_text, "\n")
    print("Anonymized TEXT:", deid_text, "\n")


if __name__ == "__main__":
    test_data_security()
