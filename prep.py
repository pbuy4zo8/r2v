import pandas as pd

import MeCab
mcb = MeCab.Tagger()

import pickle
import sys

import unicodedata

import re

def bunkatsu(text):
    # 入力を分割して名詞のみをリスト化して返す
    hinshi_list = ["一般", "固有名詞", "接尾", "特殊"]
    try:
        stand_text = unicodedata.normalize("NFKC", text)
        re_text = re.sub('[0-9]', '', stand_text)
        node_title = mcb.parseToNode(re_text)
        words_list = []
        while node_title:
            word = node_title.surface
            hinshi = node_title.feature.split(",")[0]
            hinshi2 = node_title.feature.split(",")[1]
            if len(word) == 1 and re.match('[a-z]', word) != None:
                pass
            elif hinshi == "名詞" and hinshi2 in hinshi_list:
                temp_word = str(word).lower()
                words_list.append(unicodedata.normalize("NFKC", temp_word))
            node_title = node_title.next

        return words_list
    
    except:
        return []

def prepare_df(year_num):
    # 読み込む列を指定する
    columns_list = [
        "研究課題名",
        "研究課題/領域番号",
        "キーワード",
        "研究開始時の研究の概要",
        "研究概要",
        "研究成果の概要",
        "研究実績の概要",
        "研究機関",
        "研究代表者"
    ]
    base_df = pd.read_csv("~/r2v/all_research/kaken.nii.ac.jp_2023.csv", usecols=columns_list, dtype="object")
    
    # 2023からいくつ過去まで含めるかを設定
    for i in range(year_num):
        temp_df = pd.read_csv("~/r2v/all_research/kaken.nii.ac.jp_" + str(2023-i-1)+ ".csv", usecols=columns_list, dtype="object")
        base_df = pd.concat([base_df, temp_df])
    
    # 課題番号が無ければ課題の区別が難しいため，この時点で落としておく
    base_df = base_df.dropna(subset="研究課題/領域番号")
    base_df = base_df.reset_index(drop=True)

    return base_df
