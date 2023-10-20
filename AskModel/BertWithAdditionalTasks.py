from abc import ABC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM
import json
import os
import torch.optim as optim


class Config:
    """
    用于配置模型以及其他参数
    from_path : bert模型的名称
    addi_dim : 额外添加任务的输出维度
    device : 设备
    weights : 辅助任务和mask预测loss的权重关系
    """
    def __init__(
            self,
            from_path="bert-base-chinese",
            save_path="./finetune_embedding_model/mlm/",
            addi_dim=2,
            device='cuda:1',
            vocab_size=21128,
            class_weights=[1, 1],
            batch_size=32,
            epochs=100,
            learning_rate=2e-5,
            weight_decay=0.01,
            use_addi=True,
            save_turn=10,
            require_rand=False
    ):
        self.from_path = from_path
        self.save_path = save_path
        self.addi_dim = addi_dim
        self.device = device
        self.vocab_size = vocab_size
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_addi = use_addi
        self.save_turn = save_turn
        self.require_rand = require_rand


class Res:
    def __init__(self):
        self.loss = torch.Tensor([0])
        self.addi_loss = torch.Tensor([0])
        self.bert_loss = torch.Tensor([0])
        self.logits = torch.Tensor([0])


class BertWithAdditionalTask(nn.Module):
    """
    该类给bert添加了辅助任务，用于强化bert模型对于句子结构的认知
    config : 模型配置
    """
    def __init__(
            self,
            config
    ):
        super(BertWithAdditionalTask, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(config.from_path)
        # self.base_model = torch.load("finetune_embedding_model/mlm/bert_mlm_ep_1000_6.pth")
        self.addi_head = nn.Linear(config.vocab_size, config.addi_dim)
        self.class_weights = config.class_weights
        self.loss_fn = nn.CrossEntropyLoss()
        self.config = config

    """
    前向传播函数
    input_ids : 经过tokenizer编码之后的向量
    input_labels : 对于mask的预测结果
    addi_labels : 对于辅助任务的正确标签
    addi_mask : 对于辅助任务的mask（只需要预测其中的指定部分）
    """
    def forward(self, input_ids, input_labels=None, addi_labels=None):
        if input_labels is None:
            return self.base_model(input_ids, return_dict=True, output_hidden_states=True)
        bert_output = self.base_model(input_ids=input_ids, labels=input_labels)
        bert_loss = bert_output.loss
        bert_logits = bert_output.logits
        if addi_labels is None:
            result = Res()
            result.loss = bert_loss
            result.logits = bert_logits
            result.bert_loss = bert_loss
            return result
        addi_output = self.addi_head(bert_logits)
        # 计算损失
        addi_loss = self.loss_fn(addi_output.view(-1, self.config.addi_dim), addi_labels.view(-1))
        total_loss = self.config.class_weights[0] * bert_loss + self.config.class_weights[1] * addi_loss
        result = Res()
        result.loss = total_loss
        result.logits = bert_logits
        result.addi_loss = addi_loss
        result.bert_loss = bert_loss
        return result


class TrainDataset(Dataset, ABC):
    def __init__(
            self,
            filename,
            tokenizer,
            config
    ):
        file = open(filename, "r", encoding="utf-8")
        data = json.load(file)
        file.close()
        self.tags = []
        self.sentences = []
        for doc in data:
            for tp in data[doc]:
                if tp == "Ori":
                    for pr in data[doc][tp]:
                        self.tags.append(1)
                        self.sentences.append(pr)
                else:
                    for pr in data[doc][tp]:
                        self.tags.append(0)
                        self.sentences.append(pr)
        self.config = config
        self.ori_sentences = copy.deepcopy(self.sentences)
        self.ori_tags = copy.deepcopy(self.tags)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tags) // self.config.batch_size

    def tokenize(self, x):
        return self.tokenizer(x, max_length=512, truncation=True, padding=True, return_tensors='pt')['input_ids']

    def deal_sentence(self, sentence):
        labels = {
            "原因1": 1,
            "结果1": 2,
            "原因2": 3,
            "结果2": 4,
        }
        label_list = []
        new_sentence = ""
        i = 0
        count = 0
        mark = 0
        while i < len(sentence):
            if sentence[i] == "[":
                j = i + 1
                while sentence[j][0] != "]":
                    j += 1
                label = ""
                for k in range(i + 1, j):
                    label += sentence[k]
                i = j + 1
                if label in labels:
                    count += 1
                    if count == 2:
                        count = 0
                        mark = 0
                    else:
                        mark = labels[label]
                if len(sentence[j]) > 1:
                    new_sentence += sentence[j][1:]
                    label_list.extend([mark] * len(sentence[j][1:]))
            else:
                new_sentence += sentence[i]
                label_list.append(mark)
                i += 1
        return new_sentence, label_list

    def getData(self):
        tags = self.ori_tags
        datas = []
        for i in range(len(tags)):
            if tags[i] == 1:
                tokens = self.tokenizer.decode(self.tokenizer.encode(self.ori_sentences[i])).split(" ")
                new_sentence, label_list = self.deal_sentence(tokens)
                datas.append(('语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系[MASK][MASK]。",
                              '语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系正确。",
                              1))
            else:
                tokens = self.tokenizer.decode(self.tokenizer.encode(self.ori_sentences[i])).split(" ")
                new_sentence, label_list = self.deal_sentence(tokens)
                datas.append(('语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系[MASK][MASK]。",
                              '语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系错误。",
                              0))
        return datas

    def __getitem__(self, idx):
        texts = self.sentences[: self.config.batch_size]
        tags = self.tags[: self.config.batch_size]
        inputs = []
        labels = []
        addis = []
        for i in range(len(texts)):
            if tags[i] == 1:
                tokens = self.tokenizer.decode(self.tokenizer.encode(texts[i])).split(" ")
                new_sentence, label_list = self.deal_sentence(tokens)
                inputs.append('语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系[MASK][MASK]。")
                labels.append('语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系正确。")
                addis.append(label_list)
            else:
                tokens = self.tokenizer.decode(self.tokenizer.encode(texts[i])).split(" ")
                new_sentence, label_list = self.deal_sentence(tokens)
                inputs.append('语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系[MASK][MASK]。")
                labels.append('语句：' + '"' + new_sentence[5:-5] + '"' + "因果关系错误。")
                addis.append(label_list)
        inputs = self.tokenize(inputs)
        labels = self.tokenize(labels)
        length = inputs.shape[1]
        for i in range(inputs.shape[0]):
            addis[i] = [0] * 4 + addis[i] + [0] * (length - len(addis[i]) - 4)
        addis = torch.LongTensor(addis)
        labels[inputs != 103] = -100
        batch = {"inputs": inputs, "labels": labels, "addis": addis}
        self.sentences = self.sentences[self.config.batch_size:]
        self.tags = self.tags[self.config.batch_size:]
        if not len(self):
            self.sentences = self.ori_sentences
            self.tags = self.ori_tags
        return batch


def calculate_accuracy(output, label):
    # 将标签为-100的部分排除在外
    output = output.view(-1, 1)
    label = label.view(-1, 1)
    mask = (label != -100)
    masked_output = output[mask]
    masked_label = label[mask]

    # 计算预测正确的样本数量
    correct_samples = (masked_output == masked_label).sum().item()

    # 计算准确率
    accuracy = correct_samples / len(masked_label)
    return accuracy


def train(model, train_dataloader, test_dataloader, config, logfilename):
    use_addi = config.use_addi
    require_rand = config.require_rand
    save_turn = config.save_turn
    save_path = config.save_path
    if require_rand:
        for param in model.parameters():
            if param.requires_grad:
                param.data.uniform_(-0.02, 0.02)

    best_acc = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device(config.device)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    logfile = open(logfilename, "w", encoding="utf-8")
    for cur_epc in range(int(config.epochs)):
        training_loss = 0
        train_acc = 0
        message = "Epoch: {}".format(cur_epc + 1)
        print(message)
        logfile.write(message + "\n")
        model.train()
        data_length = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['inputs'].squeeze(0).to(device)
            input_labels = batch['labels'].squeeze(0).to(device)
            addi_labels = batch["addis"].squeeze(0).to(device)
            if not use_addi:
                addi_labels = None
            result = model(input_ids=input_ids, input_labels=input_labels, addi_labels=addi_labels)
            pred = torch.argmax(result.logits, dim=-1)
            acc = calculate_accuracy(pred, input_labels)
            loss = result.loss
            addi_loss = result.addi_loss
            bert_loss = result.bert_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            training_loss += loss.item()
            message = 'Train Epoch [{}/{}], Step [{}/{}], AddiLoss : {:.4f}, BertLoss : {:.4f}, TotalLoss: {:.4f}, TrainAcc: {:.4f}'.format(
                cur_epc + 1, int(config.epochs), step + 1,
                data_length,
                addi_loss.item(),
                bert_loss.item(),
                loss.item(),
                acc)
            train_acc += acc
            logfile.write(message + "\n")
            print(message)
        message = "Train loss: {:.4f}, Train Acc: {:.4f}".format(
            training_loss / len(train_dataloader),
            train_acc / len(train_dataloader)
        )
        print(message)
        logfile.write(message + "\n")

        test_loss = 0
        model.eval()
        test_acc = 0
        data_length = len(test_dataloader)
        for step, batch in enumerate(test_dataloader):
            input_ids = batch['inputs'].squeeze(0).to(device)
            input_labels = batch['labels'].squeeze(0).to(device)
            addi_labels = batch["addis"].squeeze(0).to(device)
            if not use_addi:
                addi_labels = None
            with torch.no_grad():
                result = model(input_ids=input_ids, input_labels=input_labels, addi_labels=addi_labels)
                pred = torch.argmax(result.logits, dim=-1)
                acc = calculate_accuracy(pred, input_labels)
                loss = result.loss
                addi_loss = result.addi_loss
                bert_loss = result.bert_loss
                test_loss += loss.item()
                message = 'Test Epoch [{}/{}], Step [{}/{}], AddiLoss : {:.4f}, BertLoss : {:.4f}, TotalLoss: {:.4f}, TestAcc: {:.4f}'.format(
                    cur_epc + 1, int(config.epochs), step + 1,
                    data_length,
                    addi_loss.item(),
                    bert_loss.item(),
                    loss.item(),
                    acc)
                test_acc += acc
                logfile.write(message + "\n")
                print(message)
        message = "Test loss: {:.4f}, Test Acc: {:.4f}".format(
            test_loss / len(test_dataloader), test_acc / len(test_dataloader)
        )
        if (cur_epc + 1) % save_turn == 0:
            if test_acc > best_acc:
                best_acc = test_acc
                if os.path.exists(os.path.join(save_path, "best_model.pth")):
                    os.remove(os.path.join(save_path, "best_model.pth"))
                torch.save(model, os.path.join(save_path, "best_model.pth"))
        print(message)
        logfile.write(message + "\n")
    logfile.write("best_acc : {:.4f}".format(best_acc))
    logfile.close()


def getAcc(model, data, tokenizer, acc_filename, false_filename, device):
    """
    这是一个用来获取准确率和其他数据的函数
    model : 测试的模型
    dataset : 用于测试的数据集
    acc_filename : 准确率文件的文件名
    false_filename : 错误文件的文件名
    """
    total = 0
    true_num = 0
    T_num = 0
    F_num = 0
    TT_num = 0
    TF_num = 0
    FT_num = 0
    FF_num = 0
    false_num = 0
    acc_file = open(acc_filename, "w", encoding="utf-8")
    false_file = open(false_filename, "w", encoding="utf-8")
    for pair in data:
        total += 1
        mask_sentence, sentence, label = pair
        text_tokens = tokenizer(mask_sentence, add_special_tokens=True,
                                padding=True, return_tensors='pt')
        input = text_tokens["input_ids"].to(device)
        output = model(input)
        logits = output.logits
        pred = torch.argmax(logits, dim=-1)
        pred = pred.data.cpu().numpy().tolist()[0]
        pred_tokens = tokenizer.decode(pred[-4:-2])
        if label == 1:
            T_num += 1
            if pred_tokens[0] == "正":
                true_num += 1
                TT_num += 1
            else:
                false_num += 1
                TF_num += 1
                false_file.write(sentence + "\n")
        else:
            F_num += 1
            if pred_tokens[0] == "错":
                true_num += 1
                FT_num += 1
            else:
                false_num += 1
                FF_num += 1
                false_file.write(sentence + "\n")
    acc_file.write("acc : " + str(true_num / total) + "\n")
    acc_file.write("T : " + str(T_num) + "\n")
    acc_file.write("TT : " + str(TT_num) + "\n")
    acc_file.write("TF : " + str(TF_num) + "\n")
    acc_file.write("F : " + str(F_num) + "\n")
    acc_file.write("FF : " + str(FF_num) + "\n")
    acc_file.write("FT : " + str(FT_num) + "\n")
    acc_file.close()
    false_file.close()


if __name__ == "__main__":
    ### freedomking/mc-bert
    config = Config(epochs=100, class_weights=[0.35, 0.65],
                    device="cuda:0", save_path="./bert_base_chinese_mlm_act3_4",
                    save_turn=1, from_path="bert-base-chinese", use_addi=False)
    device = torch.device(config.device)
    model = BertWithAdditionalTask(config=config).to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(config.from_path)
    train_dataset = TrainDataset("dataset/train_3_6_1_full_deal.json", bert_tokenizer, config)
    train_dataloader = DataLoader(train_dataset)
    test_dataset = TrainDataset("dataset/test_3_6_1_full_deal.json", bert_tokenizer, config)
    test_dataloader = DataLoader(test_dataset)
    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, config=config,
          logfilename="log/bert_base_chinese_log_act3_4.txt")
    model = torch.load(os.path.join(config.save_path, "best_model.pth"))
    train_data = train_dataset.getData()
    test_data = test_dataset.getData()
    getAcc(model,
           train_data,
           bert_tokenizer,
           "acc_logFile/bert_base_chinese_mlm_acc_log_train_act3_4.txt",
           "false_logFile/bert_base_chinese_mlm_false_log_train_act3_4.txt",
           device
           )

    getAcc(model,
           test_data,
           bert_tokenizer,
           "acc_logFile/bert_base_chinese_mlm_acc_log_test_act3_4.txt",
           "false_logFile/bert_base_chinese_mlm_false_log_test_act3_4.txt",
           device
           )
