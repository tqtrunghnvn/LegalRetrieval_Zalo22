import string
import re
def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def remove_numbers(text_in):
  for ele in text_in.split():
    if ele.isdigit():
        text_in = text_in.replace(ele, "@")
  for character in text_in:
    if character.isdigit():
        text_in = text_in.replace(character, "@")
  return text_in


def remove_special_characters(text):
  chars = re.escape(string.punctuation)
  return re.sub(r'['+chars+']', '', text)


def word_segment(sent):
  sent = " ".join(rdrsegmenter.tokenize(sent.replace("\n", " ").lower())[0])
  return sent


def preprocess(text_in):
    text = clean_text(text_in)
    text = remove_special_characters(text)
    text = remove_numbers(text)
    text = word_segment(text)
    return text

# Load the model in fairseq
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.models.roberta import RobertaModel
phoBERT = RobertaModel.from_pretrained('pretrain_models/PhoBERT_base_fairseq', checkpoint_file="model.pt")

# Khởi tạo Byte Pair Encoding cho PhoBERT
class BPE():
    # bpe_codes = '/content/drive/MyDrive/NUCE/NLP/QA/BERT/fairseq/checkpoints/bpe.codes'
    bpe_codes = 'pretrain_models/PhoBERT_base_fairseq/bpe.codes'

args = BPE()
phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

import random
import re

seed = "Cho em hỏi bao giờ thì có bằng tốt nghiệp ạ"
intent = "TN"
words = preprocess(seed).split()

seed = " ".join(words)

gen_sentence = []
for i in range(len(words)):
    tmp = words[i]
    words[i] = "<mask>"
    mask = "<s>'+intent+'</s> " + ' '.join(words) + "</s>"
    print(mask)
    topk_filled_outputs = phoBERT.fill_mask(mask , topk=1)
    words[i] = tmp
    gen_sentence.append(topk_filled_outputs[0][2])

print(gen_sentence)