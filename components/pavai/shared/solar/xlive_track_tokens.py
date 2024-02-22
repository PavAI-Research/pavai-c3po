#pip install tiktoken
import tiktoken
import nltk

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")    
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens(text, model_name="gpt-3.5-turbo", debug=False):
    """
    Count the number of tokens in a given text string without using the OpenAI API.
    
    This function tries three methods in the following order:
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    1. tiktoken (preferred): Accurate token counting similar to the OpenAI API.
    2. nltk: Token counting using the Natural Language Toolkit library.
    3. split: Simple whitespace-based token counting as a fallback.
    
    Usage:
    ------
    text = "Your text here"
    result = count_tokens(text, model_name="gpt-3.5-turbo", debug=True)
    print(result)
    Required libraries:
    -------------------
    - tiktoken: Install with 'pip install tiktoken'
    - nltk: Install with 'pip install nltk'
    Parameters:
    -----------
    text : str
        The text string for which you want to count tokens.
    model_name : str, optional
        The OpenAI model for which you want to count tokens (default: "gpt-3.5-turbo").
    debug : bool, optional
        Set to True to print error messages (default: False).
    Returns:
    --------
    result : dict
        A dictionary containing the number of tokens and the method used for counting.
    """

    # Try using tiktoken
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(text))
        result = {"n_tokens": num_tokens, "method": "tiktoken"}
        return result
    except Exception as e:
        if debug:
            print(f"Error using tiktoken: {e}")
        pass

    # Try using nltk
    try:
        import nltk
        nltk.download("punkt")
        tokens = nltk.word_tokenize(text)
        result = {"n_tokens": len(tokens), "method": "nltk"}
        return result
    except Exception as e:
        if debug:
            print(f"Error using nltk: {e}")
        pass

    # If nltk and tiktoken fail, use a simple split-based method
    tokens = text.split()
    result = {"n_tokens": len(tokens), "method": "split"}
    return result

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
            new_tokens=count_tokens(record["content"], model_name=model_name, debug=debug)["n_tokens"]
        #new_tokens = count_tokens(text, model_name=model_name, debug=debug)["n_tokens"]
            self.token_count += new_tokens
            self.buffer += record["content"]
            self.token_lengths.append(new_tokens)

        # keep only half, the rest generate a summary
        if self.token_count > self.max_tokens:
            self.max_tokens = self.max_tokens/2

        while self.token_count > self.max_tokens:
            if len(history)>0:
                self.overflow_summary += self.history.pop(0)["content"]
            removed_tokens = self.token_lengths.pop(0)
            self.token_count -= removed_tokens
            self.buffer = self.buffer.split(" ", removed_tokens)[-1]
        
    def get_buffer(self):
        return self.buffer    


history=[]
history.append({'role': 'user', 'content': "1.hello world"})
history.append({'role': 'user', 'content': "2. quick brown fox jumps over the lazy dog "})
history.append({'role': 'user', 'content': "3. quick brown fox jumps over the lazy dog "})
history.append({'role': 'user', 'content': "4. quick brown fox jumps over the lazy dog "})
history.append({'role': 'user', 'content': "5. quick brown fox jumps over the lazy dog Initialize a TokenBuffer with a maximum token count of 30"})
history.append({'role': 'user', 'content': "6. quick brown fox jumps over the lazy dog Initialize a TokenBuffer with a maximum token count of 30"})
history.append({'role': 'user', 'content': "7. quick brown fox jumps over the lazy dog Initialize a TokenBuffer with a maximum token count of 30"})
history.append({'role': 'user', 'content': "8. quick brown fox jumps over the lazy dog Initialize a TokenBuffer with a maximum token count of 30"})


# history_buffer = HistoryTokenBuffer(history=history,max_tokens=120)
# history_buffer.update("I'm doing well, thank you!")
# if len(history_buffer.overflow_summary)>0:
#     print("overflow summary:",history_buffer.overflow_summary)
#     # generate summary of the overflow text
#     summary="" #llm_call()
#     history_buffer.update(summary)
# print(history_buffer.get_buffer())
# for record in history_buffer.history:
#     print(record["role"],">",record["content"])    

# print("reduced token count:", history_buffer.token_count)

"""
from token_counter import count_tokens

text = "The quick brown fox jumps over the lazy dog."
result = count_tokens(text, debug=True)
print(result)

from token_counter import TokenBuffer

# Initialize a TokenBuffer with a maximum token count of 30
buffer = TokenBuffer(max_tokens=30)

# Add a sentence to the buffer
buffer.update("Hello, how are you doing?")
print(buffer.get_buffer())
print("Token count:", buffer.token_count)

# Add another sentence to the buffer
buffer.update("I'm doing well, thank you!")
print(buffer.get_buffer())
print("Token count:", buffer.token_count)

# Add a longer sentence to the buffer
buffer.update("I've been working on a project and making great progress.")
print(buffer.get_buffer())
print("Token count:", buffer.token_count)

# Add one more sentence to the buffer
buffer.update("That's great to hear, keep up the good work!")
print(buffer.get_buffer())
print("Token count:", buffer.token_count)

"""