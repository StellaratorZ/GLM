import torch
import json
import os
from transformers import AutoTokenizer, AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    filename = "YOUR-DATA-FILE"
    base_dir = "YOUR-BASE-DIR"
    result_filename = "YOUR-RESULT-FILE"
    tokenizer = AutoTokenizer.from_pretrained(
        "./ChatGLM_Med", trust_remote_code=True)
    model = ChatGLMForConditionalGeneration.from_pretrained(
        "./ChatGLM_Med").half().cuda()
    result_record = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for idx, unit in enumerate(data):
            contents = unit["text"].split("根据以上辅助知识和你已知的知识，回答：语句")
            knowledge = contents[0]
            text = contents[1].split("因果逻辑正确还是错误")[0]
            question = "语句：" + text + "这个语句是否逻辑正确？先回答是或者否，再给出对应的理由。"
            history = []
            prompt = ""
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(0, "你现在在进行句子因果逻辑关系分析的任务。", "好的")
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(1, "可能会出现因果倒置，涉及到因果关系的对象对应关系错误等错误。", "好的")
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(2, "你现在在进行句子因果逻辑关系分析的任务。", "好的")
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(3, "这部分是为你提供的额外医疗知识：" + knowledge, "好的")
            question = prompt + "这部分是问题：" + question
            response, history = model.chat(tokenizer, question, history=history)
            result_record.append(
                {
                    "text": unit["text"],
                    "response": response
                }
            )

    with open(os.path.join(base_dir, result_filename), "w", encoding="utf-8") as f:
        f.write(json.dumps(result_record, indent=4, ensure_ascii=False))
