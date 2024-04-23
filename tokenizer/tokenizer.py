from typing import List

import tiktoken

class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def load_tokenizer(self, load_pth: str) -> None:
        try:
            self.tokenizer = Tokenizer.from_file(load_pth)
        except FileNotFoundError:
            print("File to tokenizer was not found.")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return self.tokenizer.encode_batch(texts)
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)
    
    def decode_batch(self, ids: List[List[int]]) -> List[str]:
        return self.tokenizer.decode_batch(ids)