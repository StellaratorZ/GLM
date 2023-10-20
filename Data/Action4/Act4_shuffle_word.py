import json
import random
import copy
import jieba
import random
import jieba.posseg as pseg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file",type=str)

args = parser.parse_args()
def shuffle_nouns(sentence):
    words = pseg.cut(sentence)  # Segmentation with lexical annotation

    nouns = []  # Storing lists of nouns

    for word, flag in words:
        if flag.startswith('n'): 
            nouns.append(word)

    num_nouns_to_shuffle = len(sentence)//10  # Select the number of nouns to be disrupted

    # Randomly select the index of the noun to be disrupted
    shuffle_indices = random.sample(range(len(nouns)), num_nouns_to_shuffle)

    # Disrupt the order of selected nouns
    random.shuffle(shuffle_indices)

    # Rearranging the order of nouns according to a disrupted index
    shuffled_nouns = [nouns[i] for i in range(len(nouns)) if i not in shuffle_indices]
    for i in shuffle_indices:
        shuffled_nouns.insert(i, nouns[i])

    # Combine a list of nouns and return the result
    shuffled_sentence = sentence
    for i, noun in enumerate(nouns):
        shuffled_sentence = shuffled_sentence.replace(noun, shuffled_nouns[i], 1)

    return shuffled_sentence

def shuffle_words(sentence):

    words = list(jieba.cut(sentence))

    num_words_to_shuffle = len(sentence)//7


    shuffle_indices = random.sample(range(len(words)), num_words_to_shuffle)

    random.shuffle(shuffle_indices)


    shuffled_words = [words[i] for i in range(len(words)) if i not in shuffle_indices]
    for i in shuffle_indices:
        shuffled_words.insert(i, words[i])


    shuffled_sentence = "".join(shuffled_words)
    return shuffled_sentence
def swap_words(sentence):
    
    words = list(jieba.cut(sentence))

    num_words_to_swap = len(sentence)//10
    print(num_words_to_swap)

    swap_indices = random.sample(range(len(words)), num_words_to_swap)

    words[swap_indices[0]], words[swap_indices[1]] = words[swap_indices[1]], words[swap_indices[0]]

    swapped_sentence = "".join(words)
    return swapped_sentence

with open(args.input_file, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Reordering phrases and reinserting them into sentences
result = []
count_m2=0
for data in dataset:
    count_m2+=1
    if count_m2 >1000:
        break
    text = data["text"]
    shuf_text = shuffle_words(text)

    result.append({
                "Ori": text,                        
                "Aug": shuf_text
            })
   
    result.append({
        "Ori": 15,
        "Aug": 15,    
        })    



fout = open("Act4_shuffle_word.tsv", "w")
count = 0
for idx, data in enumerate(result):
    if(data['Ori']==15):
        count+=1
        fout.write(f"doc_{count}\n") 
        continue
    fout.write("\t".join(["Ori", data['Ori']]) + "\n")
    fout.write("\t".join(["Aug", data['Aug']]) + "\n")

    
print("转换完成并保存到 Act4_shuffle_word.tsv 文件。")
