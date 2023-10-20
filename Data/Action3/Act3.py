import random
import json
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file",type=str)

args = parser.parse_args()
input_file = args.input_file
output_file = "Act3_2_rela1_co.json"

with open(input_file, "r") as f:
    dataset = json.load(f)

result2 = []


# Filter out elements with a relationship field of 3
for data in dataset:
        result = []
    # if "第一" not in text and "如下" not in text and "以下" not in text:
        head_count = {}
        tail_count= {}
        fin_data = []
        text = data["text"]
        filtered_data = [relation for relation in data["relation_of_mention"] if relation["relation"] == 1]

        if len(filtered_data)>0:
            #Counts the number of mentions under the same HEAD:
            for item in filtered_data:
                # idx = copy.deepcopy(item['head']['start_idx'])
                head_mention = item['head']['mention']+"idx"+str(item['head']['start_idx'])+"idx"+str(item['head']['end_idx'])
                if head_mention in head_count:
                    head_count[head_mention]+=1
                else:
                    head_count[head_mention] = 1
      
        
        

            for head_mention, count in head_count.items():
                if count>0:
                #Filter out groups with mentions greater than 1 under the same HEAD and merge them:    
                    tmention = []  
                    for fin_item in filtered_data:
                        
                        if fin_item['head']['mention'] == head_mention.split('idx')[0] and int(fin_item['head']['start_idx']) == int(head_mention.split('idx')[1]):
                            tmention.append(
                                {
                                "mention":fin_item["tail"]['mention'],
                                "start_idx": fin_item["tail"]['start_idx'],
                                "end_idx": fin_item["tail"]['end_idx']
                                }
                                )
                    tmentions_sorted = sorted(tmention, key=lambda x: int(x["start_idx"]))
                    start_idx = int(tmentions_sorted[0]["start_idx"])
                    end_idx = int(tmentions_sorted[-1]["end_idx"])
                    # print(text[start_idx:end_idx])
                    t_tmentions_sorted = {
                        
                                "mention":text[start_idx:end_idx],
                                "start_idx": start_idx,
                                "end_idx": end_idx     
                    }
                    h_t_mentions ={
                            "head": {
                            "mention": head_mention.split('idx')[0],
                            "start_idx":  head_mention.split('idx')[1],
                            "end_idx":  head_mention.split('idx')[2]
                            },
                            "relation": 1,
                            "tail": t_tmentions_sorted           
                    }                
                    
        
                    

                    result.append(
                    h_t_mentions
                )
            # print(result)
            for item in result:
                tail_mention = item['tail']['mention']+"idx"+str(item['tail']['start_idx'])+"idx"+str(item['tail']['end_idx'])
                
                if tail_mention in tail_count:
                    tail_count[tail_mention]+=1
                else:
                    tail_count[tail_mention] = 1
            for tail_mention, t_count in tail_count.items():
            
  
                hmention = []  
                for fin_item in result:
                    # print(tail_mention.split('idx')[1])
                    # print("tail_mention",fin_item['tail']['mention'])

                    if   (tail_mention.split('idx')[0] == fin_item['tail']['mention']) and (int(fin_item['tail']['start_idx']) == int(tail_mention.split('idx')[1])):#and (int(fin_item['tail']['end_idx']) >= int(tail_mention.split('idx')[2])):
                        hmention.append(
                            {
                            "head_mention":fin_item["head"]['mention'],
                            "h_start_idx": fin_item["head"]['start_idx'],
                            "h_end_idx": fin_item["head"]['end_idx'],
                            "tail_mention":fin_item["tail"]['mention'],
                            "t_start_idx": fin_item["tail"]['start_idx'],
                            "t_end_idx": fin_item["tail"]['end_idx']
                            }

                            )
               
           
                hmentions_sorted =sorted(hmention, key=lambda x: int(x["h_start_idx"]))
                # print("hmentions",hmention)
                # print("hmentions_sorted",hmentions_sorted)
                start_idx = int(hmentions_sorted[0]["h_start_idx"])
                end_idx = int(hmentions_sorted[-1]["h_end_idx"])
                # print(text[start_idx:end_idx])
                h_hmentions_sorted = {
                    
                            "mention":text[start_idx:end_idx],
                            "start_idx": start_idx,
                            "end_idx": end_idx     
                }
                t_h_mentions =[{
                        "head":h_hmentions_sorted,
                        "relation": 1,
                        "tail":  {
                        "mention": hmentions_sorted[0]["tail_mention"],
                        "start_idx": hmentions_sorted[0]["t_start_idx"],
                        "end_idx":  hmentions_sorted[0]["t_end_idx"]
                        }         
                }                 
                ] 
                test_t_h_mentions={
                    "text":text,
                    "relation_of_mention":t_h_mentions
                }
                if test_t_h_mentions not in result2:
                    
    
                    result2.append(test_t_h_mentions
                )
            #         fin_data.append(hmention)
            # if fin_item:
            #     result.append({
            #         "text":text,
            #         "mentions":fin_data
            #     })


with open(output_file, "w") as f:
    json.dump(result2, f, ensure_ascii=False, indent=2)


#standardized form ######################
#Pick out all the causal pairs and the long sentences in which they are located (exclude words containing '第一', '如下', and '以下')
#忽略
input_file = "Act3_2_rela1_co.json"
output_file = "Act3_3_rela1.json"

with open(input_file, "r") as f:
    dataset = json.load(f)
result = []

fin_data = []

count = 1

with open('Act2_input.json', "r") as f:
    dataset1 = json.load(f)

for data in dataset1:
    text = data["text"]
    if "第一、" not in text and "1、" not in text and "一、" not in text and "如下:" not in text and "如下几" not in text and "以下几" not in text and "(1)" not in text:
        count +=1
print(count)

for data in dataset:
    text = data["text"]
    if "第一、" not in text and "1、" not in text and "一、" not in text and "如下:" not in text and "如下几" not in text and "以下几" not in text and "(1)" not in text:
        head_count = {}
        filtered_data = [relation for relation in data["relation_of_mention"] if relation["relation"] == 1]

        for item in filtered_data:
            result.append({
                "text":text,
                "mentions": [
                {
                "mention":item["head"]['mention'],
                "start_idx": item["head"]['start_idx'],
                "end_idx": item["head"]['end_idx']
                },
                {"mention":item["tail"]['mention'],
                "start_idx": item["tail"]['start_idx'],
                "end_idx": item["tail"]['end_idx']}
                ]
            })
       
# remove sentences without relation########

with open(output_file, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)




def find_sentences_with_mentions(text, mentions):
 
    
    sentences = text.split('。')

    sentences_with_mentions = []
    text_len = 0
    for sentence in sentences:
   
        exist = 0
        for mention in mentions:
            start = text.find(sentence) 
            end = start + len(sentence)
            start_idx = mention['start_idx']
            end_idx = mention['end_idx']
            
            if start_idx >= start and  end_idx <= end and mention['mention'] in sentence:
                # if start_idx >= start and  end_idx > end and mention['mention'] in sentence:
                #     sentence = sentence+sentences(sentences.index(sentence)+1)
                exist += 1
                mention['start_idx']=text_len+(start_idx-start)   
                mention['end_idx'] = mention['start_idx']+len(mention['mention'])   
            elif start_idx >= start and start_idx <= end and end< end_idx and start!=0:# and len(mention['mention'])<64 :
                print("len(mention)",len(mention['mention']))
                print(sentence)
                print(mention)
                print("end_idx",end_idx)
                print("end",end)
                print("start_idx",start_idx)
                print("start",start)

                # print(sentence)
                # print(mention)
                # print("end_idx",end_idx)
                # print("end",end)
                # print("start_idx",start_idx)
                # print("start",start)
    
        if exist == 2:
            sentences_with_mentions.append(sentence.strip())
            text_len +=(len(sentence)+1)
               
    return sentences_with_mentions


with open('Act3_3_rela1.json', 'r') as file:
    data = json.load(file)

count = 0
for item in data:
    text = item['text']
    mentions_list = item['mentions']
    

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
    
    


    

with open('Act3_5_delet_lessmention_test.json', 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Act3_5_delet_lessmention_test已生成。")

#combine phrases####################

#Combine the phrases from the previous step to get a new long sentence
#1, first of all for Act2 filtered out of the sentence (no contextual relationship, there is mention,) for further screening (does not contain the first, as follows, etc., this type of sentence contextual relationship is relatively close), and then divided into sentences and then combined to try to construct contextual relevance Sentences with poor contextual relevance but with cause and effect and correct cause and effect.

def generate_augmented_sentences(data, num_sentences):
    augmented_sentences = []
    with open(data, "r") as f:
        dataset = json.load(f)
    
    count = 0
    filtered_sentences = [sentence for sentence in dataset if (len(sentence['text']) < 64 and len(sentence['mentions'][1]['mention'])<64 and len(sentence['mentions'][0]['mention'])<64)]

    print("len:",len(filtered_sentences))
    previous_selections = set()
    
    while count<num_sentences:
        random_indices = random.sample(range(len(filtered_sentences)), 2)

        idx1 = random_indices[0]
        idx2 = random_indices[1]


        index1 = filtered_sentences[idx1]['text']
        index2 = filtered_sentences[idx2]['text']
        if (idx1, idx2) not in previous_selections and (idx2, idx1) not in previous_selections and index1!=index2 and  index1 and index2:
            count+=1

            previous_selections.add((idx1, idx2))
            ori1 = filtered_sentences[idx1]['text']
            ori2 = filtered_sentences[idx2]['text']
            augmented_sentence = ori1 + "[SEP]" + ori2
            mentions1 =filtered_sentences[idx1]['mentions']
            mentions2 =copy.deepcopy(  filtered_sentences[idx2]['mentions'])
            text_len = len(ori1)+5
            mentions2[1]['start_idx']+=text_len
            mentions2[1]['end_idx']+=text_len
            mentions2[0]['start_idx']+=text_len
            mentions2[0]['end_idx']+=text_len
            
            mentions12 = mentions1+mentions2

            augmented_sentences.append({
                "text":augmented_sentence,
                "mentions": mentions12
                }
                )

    return augmented_sentences

    



data_file = 'Act3_5_delet_lessmention_test.json'
output_file = 'Act3_6_comb.json'  

num_augmented_sentences = 10000

augmented_sentences = generate_augmented_sentences(data_file, num_augmented_sentences)


with open(output_file, "w") as f:
    json.dump(augmented_sentences, f, ensure_ascii=False, indent=2)


print("Act3_6_comb已生成")


#shuffle
with open('Act3_6_comb.json', 'r', encoding='utf-8') as file:
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
        mentions_mess[0],mentions_mess[2] = mentions_mess[2],mentions_mess[0]
        
        
        aug_text = text[:mentions_sorted[0]["start_idx"]]          
        # Swap the position of each mention in the original sentence
        for i in range(len(mentions_mess)):
            mention_mess = mentions_mess[i]
            mention_sorted = mentions_sorted[i]

            if i <(len(mentions_mess)-1):
                aug_text = aug_text + mention_mess["mention"] + text[mention_sorted["end_idx"]:mentions_sorted[i+1]["start_idx"]]
        aug_text = aug_text+mention_mess["mention"]+text[mention_sorted["end_idx"]:]

        
        mentions_mess2 = copy.deepcopy(mentions_sorted)
        mentions_mess2[1],mentions_mess2[3] = mentions_mess2[3],mentions_mess2[1]
        
        # Fill in the sentences with the scrambled mentions.
        aug_text2 = text[:mentions_sorted[0]["start_idx"]]          
        # Swap the position of each mention in the original sentence
        for i in range(len(mentions_mess2)):
            mention_mess = mentions_mess2[i]
            mention_sorted = mentions_sorted[i]

            if i <(len(mentions_mess2)-1):
                aug_text2 = aug_text2 + mention_mess["mention"] + text[mention_sorted["end_idx"]:mentions_sorted[i+1]["start_idx"]]
        aug_text2 = aug_text2+mention_mess["mention"]+text[mention_sorted["end_idx"]:]
       

    
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



outputfile = 'Act3_6_shuffle2.tsv'
fout = open(outputfile, "w")
count = 0
for idx, data in enumerate(result):
    if(data['Ori']==15):
        count+=1
        fout.write(f"doc_{count}\n") 
        continue
    fout.write("\t".join(["Ori", data['Ori']]) + "\n")
    fout.write("\t".join(["Aug", data['Aug']]) + "\n")

    
print("转换完成并保存到文件。"+outputfile)
