#pip install tiktoken
import tiktoken
#import nltk
import traceback

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")    
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens(text, model_name="gpt-3.5-turbo", debug=False):
    """
    Count the number of tokens in a given text string without using the OpenAI API.
    """
    if text is None:
        print("*Empty Text* nothing to count for token.")
        return 
    # Try using tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(text))
        result = {"n_tokens": num_tokens, "method": "tiktoken"}
        return result
    except Exception as e:
        print("an error occurred ", e)
        print(traceback.format_exc())
        print(f"Error using tiktoken:",e)

    # If nltk and tiktoken fail, use a simple split-based method
    if isinstance(text,str):
        tokens = text.split()
        return {"n_tokens": len(tokens), "method": "split"}
    else:
        return {"n_tokens": len(text), "method": "split"}

class TokenBuffer:
    def __init__(self, max_tokens=2048):
        self.max_tokens = max_tokens
        self.buffer = ""
        self.token_lengths = []
        self.token_count = 0

    def update(self, text, model_name="gpt-3.5-turbo", debug=False):
        new_tokens = count_tokens(text, model_name=model_name, debug=debug)["n_tokens"]
        self.token_count += new_tokens
        self.buffer += text
        self.token_lengths.append(new_tokens)

        while self.token_count > self.max_tokens:
            removed_tokens = self.token_lengths.pop(0)
            self.token_count -= removed_tokens
            self.buffer = self.buffer.split(" ", removed_tokens)[-1]

    def get_buffer(self):
        return self.buffer
    
class HistoryTokenBuffer:

    def __init__(self, history:list=[], max_tokens:int=4096*2):
        self.max_tokens = max_tokens
        self.history = history
        self.overflow_summary = ""       
        self.token_lengths = []
        self.token_count = 0
        self.buffer=""

    def update(self, text, model_name="gpt-3.5-turbo", debug=False):
        # loop the history
        for record in self.history:
            ## Note: for tool calling first call content is empty
            if record["content"] is not None:
                ## ignore non-text object??
                if isinstance(record["content"], str):
                    new_tokens=count_tokens(record["content"], model_name=model_name, debug=debug)["n_tokens"]                    
                    self.token_count += new_tokens                    
                    self.buffer += record["content"]
                    self.token_lengths.append(new_tokens)

        # keep only half, the rest generate a summary
        if self.token_count > self.max_tokens:
            self.max_tokens = self.max_tokens/2

        while self.token_count > self.max_tokens:
            if len(self.history)>0:
                self.overflow_summary += self.history.pop(0)["content"]
            removed_tokens = self.token_lengths.pop(0)
            self.token_count -= removed_tokens
            self.buffer = self.buffer.split(" ", removed_tokens)[-1]
        
    def get_buffer(self):
        return self.buffer    

