import json
import random
import copy
import argparse

#Filter and merge cause and effect: e.g. A-X A-Y B-X B-Y merged into AB-XY
parser = argparse.ArgumentParser()
parser.add_argument("--input_file",type=str)

args = parser.parse_args()

with open(args.input_file, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

result = []
mention_set = set()
conut =0
for data in dataset:
    text = data["text"]
    mentions = []

    for item in data["relation_of_mention"]:
        if item["relation"] ==1:

            if "mention" in item["head"]:
                mention = item["head"]["mention"]
                start_idx = item["head"]["start_idx"]
                end_idx = item["head"]["end_idx"]

                if (mention, start_idx, end_idx) not in mention_set:
                    mentions.append({
                        "mention": mention,
                        "start_idx": start_idx,
                        "end_idx": end_idx
                    })
                    mention_set.add((mention, start_idx, end_idx))

            if "mention" in item["tail"]:
                mention = item["tail"]["mention"]
                start_idx = item["tail"]["start_idx"]
                end_idx = item["tail"]["end_idx"]

                if (mention, start_idx, end_idx) not in mention_set:
                    mentions.append({
                        "mention": mention,
                        "start_idx": start_idx,
                        "end_idx": end_idx
                    })
                    mention_set.add((mention, start_idx, end_idx))
        elif item["relation"] ==2:
            if "mention" in item["tail"]["head"]:
                mention = item["tail"]["head"]["mention"]
                start_idx = item["tail"]["head"]["start_idx"]
                end_idx = item["tail"]["head"]["end_idx"]

                if (mention, start_idx, end_idx) not in mention_set:
                    mentions.append({
                        "mention": mention,
                        "start_idx": start_idx,
                        "end_idx": end_idx
                    })
                    mention_set.add((mention, start_idx, end_idx))

            if "mention" in item["tail"]["tail"]:
                mention = item["tail"]["tail"]["mention"]
                start_idx = item["tail"]["tail"]["start_idx"]
                end_idx = item["tail"]["tail"]["end_idx"]

                if (mention, start_idx, end_idx) not in mention_set:
                    mentions.append({
                        "mention": mention,
                        "start_idx": start_idx,
                        "end_idx": end_idx
                    })
                    mention_set.add((mention, start_idx, end_idx))

            
    result.append({
        "text": text,
        "mentions": mentions
    })


with open('Act2_1_withoutrelation.json', 'w', encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False, indent=4)

print("转换完成并保存到Act2_1_withoutrelation.json文件。")
#removeoverlap
#There are about 20 pairs of nested mentions in the original dataset, of which only one pair is a verb-object structured phrase, the others are noun phrases. Therefore for nested phrases, the smallest substring is taken and the long string is removed.
def check_overlap(mention1, mention2):
    """
    Checking for overlap between two mentions
    """
    start1, end1 = mention1['start_idx'], mention1['end_idx']
    start2, end2 = mention2['start_idx'], mention2['end_idx']
    return start1 <= start2 <= end1 or start2 <= start1 <= end2

def get_mention_length(mention):
    """
    Calculate the length of the reference
    """
    start_idx = mention['start_idx']
    end_idx = mention['end_idx']
    return end_idx - start_idx

def remove_nested_mentions(mentions):
    """
    Remove longer mentions from nested or overlapping mentions
    """
    mentions.sort(key=get_mention_length)
    removed_mentions = []
    for i in range(len(mentions)):
        mention1 = mentions[i]
        for j in range(i+1, len(mentions)):
            mention2 = mentions[j]
            if check_overlap(mention1, mention2):
                removed_mentions.append(mention2)
    return [mention for mention in mentions if mention not in removed_mentions]


with open('Act2_1_withoutrelation.json','r', encoding="utf-8") as file:
    data = json.load(file)


for item in data:
    mentions = item['mentions']
    filtered_mentions = remove_nested_mentions(mentions)
    item['mentions'] = filtered_mentions

with open('Act2_2_withoutoverlap.json', 'w', encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Act2_2_withoutoverlap.json已生成。")

#delete sentences without mention 

def find_sentences_with_mentions(text, mentions):
    # Splitting text into lists of sentences
    sentences = text.split('。')

    sentences_with_mentions = []
    text_len = 0
    for sentence in sentences:
        # Check for the presence of any mentions in the sentence
        exist = 0
        for mention in mentions:
            start = text.find(sentence)  # Determine the starting position of the substring
            end = start + len(sentence)
            start_idx = mention['start_idx']
            end_idx = mention['end_idx']
            
            if start_idx >= start and end_idx <= end and mention['mention'] in sentence:
                exist = 1
                mention['start_idx']=text_len+(start_idx-start)   
                mention['end_idx'] = mention['start_idx']+len(mention['mention'])   
        if exist == 1:
            sentences_with_mentions.append(sentence.strip())
            text_len +=(len(sentence)+1)
               
    return sentences_with_mentions


with open('Act2_2_withoutoverlap.json', 'r') as file:
    data = json.load(file)

# Update the text field for each item
count = 0
for item in data:
    text = item['text']
    mentions_list = item['mentions']
    
    #mentions de-emphasize
    mentions_update =mentions_list
    mentions_unique = set()
    mentions_update = []
    for iter in mentions_list:
        identifier = (iter['mention'],iter['start_idx'],iter['end_idx'])
        if identifier not in mentions_unique:
            mentions_unique.add(identifier)
            mentions_update.append(iter)
           

    sentences_with_mentions = find_sentences_with_mentions(text, mentions_update)
    updated_text = '。'.join(sentences_with_mentions)
    item['text'] = updated_text
    item['mentions'] = mentions_update
    
    


    
# 
with open('Act2_3_deletlessmention.json', 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Act2_3_deletlessmention已生成。")

# shuffle


with open('Act2_3_deletlessmention.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Reordering phrases and reinserting them into sentences
result = []
count_m2=0
for data in dataset:
    text = data["text"]
    mentions = data["mentions"]
   
    if (len(mentions)>2):
 
        mentions_sorted = sorted(mentions, key=lambda x: x["start_idx"])
      

        mentions_mess = copy.deepcopy(mentions_sorted)
        
        random.shuffle(mentions_mess)
        while mentions_mess==mentions:
            random.shuffle(mentions_mess)
        
        # Fill in the sentences with the scrambled mentions.
        aug_text = text[:mentions_sorted[0]["start_idx"]]          
        # Swap the position of each mention in the original sentence
        for i in range(len(mentions_mess)):
            mention_mess = mentions_mess[i]
            mention_sorted = mentions_sorted[i]
            mention_mess["mention"] = mention_mess["mention"]
# mention_mess["mention"] = "[MEN_S]"+mention_mess["mention"]+"[MEN_E]"
            if i <(len(mentions_mess)-1):
                aug_text = aug_text + mention_mess["mention"] + text[mention_sorted["end_idx"]:mentions_sorted[i+1]["start_idx"]]
        aug_text = aug_text+mention_mess["mention"]+text[mention_sorted["end_idx"]:]

        
        mentions_mess2 = copy.deepcopy(mentions_sorted)
        #Disordering to prevent the same pulling disorder result
        while(1):
            random.shuffle(mentions_mess2)
            if (mentions_mess2!=mentions_mess) and (mentions_mess2!=mentions):
                break
        
        aug_text2 = text[:mentions_sorted[0]["start_idx"]]          
        
        for i in range(len(mentions_mess2)):
            mention_mess = mentions_mess2[i]
            mention_sorted = mentions_sorted[i]

            if i <(len(mentions_mess2)-1):
                aug_text2 = aug_text2 + mention_mess["mention"] + text[mention_sorted["end_idx"]:mentions_sorted[i+1]["start_idx"]]
        aug_text2 = aug_text2+mention_mess["mention"]+text[mention_sorted["end_idx"]:]
        if "第一" not in text and "如下" not in text and "以下" not in text:
            Ori0_list = text.split('。')
            Aug1_list = aug_text.split('。')   
            Aug2_list = aug_text2.split('。')  

            for i in range(len(Ori0_list)):
                if (Ori0_list[i]!=Aug1_list[i]):
                    result.append({
                        "Ori": Ori0_list[i],
                        "Aug": Aug1_list[i],
                        # "Aug2": Aug2_list[i]
                    })
                if(Ori0_list[i]!=Aug2_list[i]):
                    result.append({
                        "Ori": Ori0_list[i],                        
                        "Aug": Aug2_list[i]
                    })
            result.append({
                "Ori": 15,
                "Aug": 15,    
                })    
        else:
            result.append({
                        "Ori": text,                        
                        "Aug": aug_text
                    })
            result.append({
                        "Ori": text,                        
                        "Aug": aug_text2
                    })
            result.append({
                "Ori": 15,
                "Aug": 15,    
                })    
    else:
        count_m2+=1
print(count_m2)



fout = open("Act2_4_shffle.tsv", "w")
count = 0
for idx, data in enumerate(result):
    if(data['Ori']==15):
        count+=1
        fout.write(f"doc_{count}\n") 
        continue
    fout.write("\t".join(["Ori", data['Ori']]) + "\n")
    fout.write("\t".join(["Aug", data['Aug']]) + "\n")

    
print("转换完成并保存到 Act2_4_shffle.tsv 文件。")