import json
import random
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file",type=str)

args = parser.parse_args()
with open(args.input_file, 'r', encoding='utf-8') as file:
    dataset = json.load(file)


result = []
mentions_all = []
for data in dataset:
    text = data["text"]
    mentions_all += data["mentions"]
    random.shuffle(mentions_all)
    mentions_all2 = copy.deepcopy(mentions_all)
    random.shuffle(mentions_all2)
mentions_num = 0
mentions_num2 = 0
for data in dataset:
    text = data["text"]
    mentions = data["mentions"]
    if mentions:
     
        mentions_sorted = sorted(mentions, key=lambda x: x["start_idx"])

        if mentions_sorted:
            aug_text = text[:mentions_sorted[0]["start_idx"]]
            
            for i in range(len(mentions_sorted)):

                mention_mess = mentions_all[mentions_num]
                mention_sorted = mentions_sorted[i]

                if i < (len(mentions_sorted)-1):
                    aug_text = aug_text + \
                        mention_mess["mention"] + text[mention_sorted["end_idx"]
                            :mentions_sorted[i+1]["start_idx"]]
                mentions_num += 1
            aug_text = aug_text + \
                mention_mess["mention"]+text[mention_sorted["end_idx"]:]

        
        if mentions_sorted:
            aug_text2 = text[:mentions_sorted[0]["start_idx"]]
            # Swap the position of each mention in the original sentence
            for i in range(len(mentions_sorted)):
                mention_mess2 = mentions_all2[mentions_num2]
                mention_sorted = mentions_sorted[i]

                if i < (len(mentions_sorted)-1):
                    aug_text2 = aug_text2 + \
                        mention_mess2["mention"] + text[mention_sorted["end_idx"]
                            :mentions_sorted[i+1]["start_idx"]]
                mentions_num2 += 1
            aug_text2 = aug_text2 + \
                mention_mess2["mention"]+text[mention_sorted["end_idx"]:]

            Ori0_list = text.split('。')
            Aug1_list = aug_text.split('。')
            Aug2_list = aug_text2.split('。')

            for i in range(len(Ori0_list)):
                result.append({
                    "Ori": Ori0_list[i],
                    "Aug": Aug1_list[i],
                    # "Aug2": Aug2_list[i]
                })
                result.append({
                    "Ori": Ori0_list[i],
                    "Aug": Aug2_list[i]
                })
            result.append({
                "Ori": 15,
                "Aug": 15,
            })



fout = open("Act4_shuffle_mention.tsv", "w")
count = 0
for idx, data in enumerate(result):
    if(data['Ori']==15):
        count+=1
        fout.write(f"doc_{count}\n") 
        continue
    fout.write("\t".join(["Ori", data['Ori']]) + "\n")
    fout.write("\t".join(["Aug", data['Aug']]) + "\n")
   

print("转换完成并保存到 Act4_shuffle_mention.tsv 文件。")
