import json
from logging import getLogger

from tqdm import tqdm

logger = getLogger()


def indexes_expand(st: int, ed: int, indexes: list):
    """
    expand the corresponding indexes part to whole sentences
    :param st:
    :param ed:
    :param indexes:
    :return: whole index in original sentence
    """
    trap = "trap"
    ret_st = indexes[0]
    ret_ed = indexes[-1]
    for index in indexes:
        if index >= st:
            break
        else:
            ret_st = index
    for index in reversed(indexes):
        if index < ed:
            break
        else:
            ret_ed = index
    return ret_st, ret_ed


def find_sentence_part_for_relas(rela1: dict, rela2: dict, indexes: list):
    """
    index updater
    judge which part of sentence should be selected to replace and make negative instances.
    More consciesly, using startidx and endidx of each rela, and extend them to find the minimum sentence(splitted by "。") to cover thme and strip unrelated parts of sentence.
    :param rela1:
    :param rela2:
    :param indexes: [0,5,1114,55554] the indexes for evrey chinese "commma"
    :return:



    """
    rela_indexs = [rela1["start_idx"], rela1["end_idx"], rela2["start_idx"], rela2["end_idx"]]
    max_rela_idx = max(rela_indexs)
    min_rela_idx = min(rela_indexs)
    return indexes_expand(min_rela_idx, max_rela_idx, indexes)


def string_find_all(string: str, to_find: str):
    ret = []
    find_res = string.find(to_find)
    while find_res != -1:
        ret.append(find_res)
        find_res = string.find(to_find, find_res + 1)
    return ret


def string_replace(string: str, head: str, tail: str):
    """
    交换head与tail的位置，来产生新的字符串
    :param string:
    :param head:
    :param tail:
    :return:
    """
    head_pos = string.find(head)

    if head_pos < 0:
        logger.error("没有找到对应的字符串，替换取消")
        return string
    part1 = string[0:head_pos]
    part2 = tail
    part3_unrep = string[head_pos + len(head):]
    part3 = part3_unrep.replace(tail, head)
    return "".join([part1, part2, part3])


def handle_relation(relation: dict):
    """
    递归地处理所有的relation，采用深度优先的方式遍历所有的链条
    所有子递归的得数全部extend过来
    :param relation:
    :return:
    """
    if relation["tail"].get("type", "mention") == "relation":
        if relation["relation"] == 1:
            ret = [(relation["head"], relation["tail"]["head"])]
        else:
            ret = []
        ret.extend(handle_relation(relation["tail"]))
        return ret
    else:
        if relation["relation"] != 3:
            return [(relation["head"], relation["tail"])]
        else:
            return []


if __name__ == "__main__":
    """
    Only for summary the amount of data
    """
    mentions = set()

    with open("../train_0717.json", "r", encoding="utf-8", errors="ignore") as fin:
        data_s = json.load(fin)

    now_count = 0
    qiantao_count = 0
    # fout = open("act_1d2_output_full@0517.tsv", "w")

    for ob in tqdm(data_s):
        texts_original = ob["text"]
        indexes = [0]
        indexes.extend(string_find_all(texts_original, "。"))
        indexes.append(len(texts_original) + (1 if texts_original[-1] else 0))
        text_new = texts_original
        if len(ob["relation_of_mention"]) == 0:
            text_new = "此句子无标注，故无法产生负例"
            continue
        relations = []
        for relation in ob["relation_of_mention"]:
            relass = handle_relation(relation)
            for (re1, re2) in relass:
                mentions.add(re1["mention"])
                mentions.add(re2["mention"])
    trap = "let us be a trap"
