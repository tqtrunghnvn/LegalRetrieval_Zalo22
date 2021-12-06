import json
from nltk.tokenize import RegexpTokenizer

file_path = ''
tokenizer = RegexpTokenizer(r'\w+')
# text = "This is my text. It icludes commas, question marks? and other stuff. Also U.S.."
text = "Thông tư này hướng dẫn tuần tra. A B C."
tokens = tokenizer.tokenize(text)
print(len(tokens), tokens)