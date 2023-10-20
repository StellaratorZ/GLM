import openai
import json
import time
import concurrent.futures


openai.api_key = "YOUR-API-KEY"
openai.organization = "YOUR-ORG"
model_name = "gpt-3.5-turbo" # "gpt-4"

def ask_gpt_model(sen_in, knowledge):
    question_sentence = "这部分是问题：语句：" + '"' + sen_in + '"' + "这个语句是否逻辑正确？先回答是或者否，再给出对应的理由"
    knowledge_sentence = "这部分是为你提供的额外医疗知识：额外医疗知识为：" + '"' + knowledge + '"'
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=1024,
                temperature=0,
                messages=
                [{"role": "system", "content": "你现在在进行句子因果逻辑关系分析的任务。"},
                 {"role": "system", "content": "可能会出现因果倒置，涉及到因果关系的对象对应关系错误等错误。"},
                 {"role": "system", "content": knowledge_sentence},
                 {"role": "user", "content": question_sentence}]
            )
            response = completion.choices[0].message['content']
            if response != "":
                return {
                    "text": sen_in,
                    "response": response
                }
        except:
            continue


if __name__ == "__main__":
    result_record = []
    result_file = "YOUR-RESULT-FILE"
    data_file = "YOUR-DATA-FILE"

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        future_to_line = {executor.submit(ask_gpt_model, line["text"], "".join(
            [h + "会导致" + t + "。" for num, (h, t) in enumerate(line["mentions"]) if num < 20])): line for line in
                          data}
        for future in concurrent.futures.as_completed(future_to_line):
            line = future_to_line[future]
            try:
                response = future.result()
                result_record.append(response)
                print(response)
            except Exception as exc:
                print(f'{line} generated an exception: {exc}')

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=4, ensure_ascii=False)
