import json


label_file = "YOUR-LABEL-FILE"
judge_file = "YOUR-JUDGE-FILE"
origin_file = "YOUR-ORI-FILE"


def judge_label(sentence):
    if "不正确" in sentence or "不是" in sentence or "错误" in sentence or "否" in sentence:
        return False
    return True


true_label = []
with open(label_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    for i in data:
        true_label.append(i["label"])


judge_data = []
with open(origin_file, "r", encoding="utf-8") as f:
    index = 0
    data = json.load(f)
    for i in data:
        label = true_label[index]
        index += 1
        response = i["response"]
        predict_label = judge_label(response)
        judge_data.append(
            {
                "text": i["text"],
                "response": i["response"],
                "label": label,
                "predict_label": int(predict_label == 1)
            }
        )
    with open(judge_file, "w", encoding="utf-8") as j_file:
        json.dump(judge_data, j_file, indent=4, ensure_ascii=False)
