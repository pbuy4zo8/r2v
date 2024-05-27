'''
    written by K.Kobayashi(jeffy890)
    2024.05.16

    r2v
'''
import pandas as pd
import numpy as np
import scipy as sp

import MeCab
mcb = MeCab.Tagger()

import pickle
import sys

import unicodedata

import re

from prep import *


def pick_orgs():
    year_num = 20
    base_df = prepare_df(year_num)
    print("preparing dataframes done")
    print(str(len(base_df)) + " num of index will be used")

    drop_list = []
    for index, row in base_df.iterrows():
        temp_kikan = row["研究機関"]
        try:
            if "岡山大学" not in temp_kikan:
                drop_list.append(index)
            
        except:
            drop_list.append(index)

        if index % 5000 == 0:
            print("#", end="", flush=True)

        '''
        if index > 20000:
            break
        '''
    print("")
    base_df = base_df.drop(drop_list)

    base_df = base_df.drop_duplicates(keep="last", subset="研究課題/領域番号")
    base_df = base_df.reset_index(drop=True)

    base_df.to_csv("./kaken_picked.csv", encoding="utf_8_sig", index=False)
    print(str(len(base_df)) + " num of index picked")
    

def create_dict():
    base_df = pd.read_csv("./kaken_picked.csv")
    print(len(base_df))

    # keywordは別で判定する．そのための辞書を宣言する
    key_dict = {}
    id_to_key = {}
    key_dict_count = 0

    word_dict = {}
    id_to_word = {}
    word_dict_count = 0
    for index, row in base_df.iterrows():
        words_list = []
        kadai_title = row["研究課題名"]
        kaishi_gaiyou = row["研究開始時の研究の概要"]
        keywords = row["キーワード"]

        # keyword判定
        try:
            keywords_hantei = keywords.split(" / ")
            for i in range(len(keywords_hantei)):
                if keywords_hantei[i] not in key_dict:
                    key_dict[keywords_hantei[i]] = key_dict_count
                    id_to_key[key_dict_count] = keywords_hantei[i]
                    key_dict_count += 1
        except:
            pass

        words_list.extend(bunkatsu(keywords))
        words_list.extend(bunkatsu(kadai_title))
        words_list.extend(bunkatsu(kaishi_gaiyou))
        words_list.extend(bunkatsu(row["研究概要"]))
        words_list.extend(bunkatsu(row["研究成果の概要"]))
        words_list.extend(bunkatsu(row["研究実績の概要"]))

        words_list = list(set(words_list))

        for i in range(len(words_list)):
            if words_list[i] not in word_dict:
                word_dict[words_list[i]] = word_dict_count
                id_to_word[word_dict_count] = words_list[i]
                word_dict_count += 1

        if index % 5000 == 0:
            print("#", end="", flush=True)


    with open("./vectors/key_dict.pkl", "wb") as f:
        pickle.dump(key_dict, f)
    with open("./vectors/id_to_key.pkl", "wb") as f:
        pickle.dump(id_to_key, f)

    # dictを保存
    with open("./vectors/word_dict.pkl", "wb") as f:
        pickle.dump(word_dict, f)
    with open("./vectors/id_to_word.pkl", "wb") as f:
        pickle.dump(id_to_word, f)

    print("\ndict preparing done")
    print("dict len (word): " + str(len(word_dict)))
    print("dict len (key) : " + str(len(key_dict)))

def pre_count():
    with open("./vectors/word_dict.pkl", "rb") as f:
        word_dict = pickle.load(f)
    with open("./vectors/id_to_word.pkl", "rb") as f:
        id_to_word = pickle.load(f)

    with open("./vectors/key_dict.pkl", "rb") as f:
        key_dict = pickle.load(f)
    with open("./vectors/id_to_key.pkl", "rb") as f:
        id_to_key = pickle.load(f)

    base_df = pd.read_csv("./kaken_picked.csv")

    # 辞書を用いた単語カウント
    # ベクトル用arrayの宣言
    # word_vector = np.zeros((len(base_df), len(word_dict)), dtype="float32")
    tate_sum = np.zeros((1, len(word_dict)+len(key_dict)), dtype="uint32")

    for index, row in base_df.iterrows():
        words_list = []
        kadai_title = row["研究課題名"]
        kaishi_gaiyou = row["研究開始時の研究の概要"]
        keywords = row["キーワード"]

        try:
            keywords_hantei = keywords.split(" / ")
            for i in range(len(keywords_hantei)):
                if keywords_hantei[i] in key_dict:
                    tate_sum[0][len(word_dict)+key_dict[keywords_hantei[i]]] += 1
        except:
            pass

        for i in range(len(key_dict)):
            try:
                if id_to_key[i] in kadai_title:
                    id_to_key[0][len(word_dict)+i] += 1
            except:
                pass
            
            try:
                if id_to_key[i] in kaishi_gaiyou:
                    id_to_key[0][len(word_dict)+i] += 1
                    
            except:
                pass

            try:
                if id_to_key[i] in row["研究概要"]:
                    id_to_key[0][len(word_dict)+i] += 1
            except:
                pass

            try:
                if id_to_key[i] in row["研究成果の概要"]:
                    id_to_key[0][len(word_dict)+i] += 1
            except:
                pass

            try:
                if id_to_key[i] in row["研究実績の概要"]:
                    id_to_key[0][len(word_dict)+i] += 1
            except:
                pass
        
        words_list.extend(bunkatsu(keywords))
        words_list.extend(bunkatsu(kadai_title))
        words_list.extend(bunkatsu(kaishi_gaiyou))
        words_list.extend(bunkatsu(row["研究概要"]))
        words_list.extend(bunkatsu(row["研究成果の概要"]))
        words_list.extend(bunkatsu(row["研究実績の概要"]))

        for i in range(len(words_list)):
            if words_list[i] in word_dict:
                tate_sum[0][word_dict[words_list[i]]] += 1
        
        if index % 5000 == 0:
            print("#", end="", flush=True)

        print("done")
        print(keywords)
        sys.exit()
        
    np.save("./vectors/tate_sum.npy", tate_sum)

    dict_count = 0
    red_word_dict = {}
    red_id_to_word = {}
    for i in range(len(word_dict)+len(key_dict)):
        if tate_sum[0][i] < 10:
            continue
        else:
            red_word_dict[id_to_word[i]] = dict_count
            red_id_to_word[dict_count] = id_to_word[i]
            dict_count += 1

    print(len(red_word_dict))

    with open("./vectors/red_word_dict.pkl", "wb") as f:
        pickle.dump(red_word_dict, f)
    with open("./vectors/red_id_to_word.pkl", "wb") as f:
        pickle.dump(red_id_to_word, f)

    np.savetxt("./sum_of_tate.csv", tate_sum, delimiter=",")

# 辞書の作成とvectorへのカウント
def pre_calc():
    with open("./vectors/word_dict.pkl", "rb") as f:
        word_dict = pickle.load(f)
    with open("./vectors/id_to_word.pkl", "rb") as f:
        id_to_word = pickle.load(f)

    with open("./vectors/key_dict.pkl", "rb") as f:
        key_dict = pickle.load(f)
    with open("./vectors/id_to_key.pkl", "rb") as f:
        id_to_key = pickle.load(f)

    base_df = pd.read_csv("./kaken_picked.csv")

    # 辞書を用いた単語カウント
    # ベクトル用arrayの宣言
    word_vector = np.zeros((len(base_df), len(word_dict)), dtype="float32")

    # カウント
    for index, row in base_df.iterrows():
        words_list = []
        kadai_title = row["研究課題名"]
        kaishi_gaiyou = row["研究開始時の研究の概要"]
        keywords = row["キーワード"]

        try:
            keywords_hantei = keywords.split(" / ")
            for i in range(len(keywords_hantei)):
                if keywords_hantei[i] in key_dict:
                    word_vector[index][len(word_dict)+key_dict[keywords_hantei[i]]] += 1
        except:
            pass

        for i in range(len(key_dict)):
            try:
                if id_to_key[i] in kadai_title:
                    word_vector[index][len(word_dict)+i] += 1
            except:
                pass
            
            try:
                if id_to_key[i] in kaishi_gaiyou:
                    word_vector[index][len(word_dict)+i] += 1
            except:
                pass

            try:
                if id_to_key[i] in row["研究概要"]:
                    word_vector[index][len(word_dict)+i] += 1
            except:
                pass

            try:
                if id_to_key[i] in row["研究成果の概要"]:
                    word_vector[index][len(word_dict)+i] += 1
            except:
                pass

            try:
                if id_to_key[i] in row["研究実績の概要"]:
                    word_vector[index][len(word_dict)+i] += 1
            except:
                pass

        words_list.extend(bunkatsu(keywords))
        words_list.extend(bunkatsu(kadai_title))
        words_list.extend(bunkatsu(kaishi_gaiyou))
        words_list.extend(bunkatsu(row["研究概要"]))
        words_list.extend(bunkatsu(row["研究成果の概要"]))
        words_list.extend(bunkatsu(row["研究実績の概要"]))

        for i in range(len(words_list)):
            if words_list[i] in word_dict:
                word_vector[index][word_dict[words_list[i]]] += 1

        if index % 50 == 0:
            print("#", end="", flush=True)

    print("\nword_vector count done")
    np.save("./vectors/word_vector.npy", word_vector)

def pre_calc_red():
    with open("./vectors/red_word_dict.pkl", "rb") as f:
        word_dict = pickle.load(f)
    with open("./vectors/red_id_to_word.pkl", "rb") as f:
        id_to_word = pickle.load(f)

    base_df = pd.read_csv("./kaken_picked.csv")

    # 辞書を用いた単語カウント
    # ベクトル用arrayの宣言
    word_vector = np.zeros((len(base_df), len(word_dict)), dtype="float32")

    # カウント
    for index, row in base_df.iterrows():
        words_list = []
        kadai_title = row["研究課題名"]
        kaishi_gaiyou = row["研究開始時の研究の概要"]
        keywords = row["キーワード"]
        
        words_list.extend(bunkatsu(keywords))
        words_list.extend(bunkatsu(kadai_title))
        words_list.extend(bunkatsu(kaishi_gaiyou))
        words_list.extend(bunkatsu(row["研究概要"]))
        words_list.extend(bunkatsu(row["研究成果の概要"]))
        words_list.extend(bunkatsu(row["研究実績の概要"]))

        for i in range(len(words_list)):
            if words_list[i] in word_dict:
                word_vector[index][word_dict[words_list[i]]] += 1

        if index % 5000 == 0:
            print("#", end="", flush=True)

    print("\nword_vector count done")
    np.save("./vectors/red_word_vector.npy", word_vector)

def dict_analysis():
    word_vector = np.load("./vectors/word_vector.npy")
    print(word_vector.shape)
    print(sys.getsizeof(word_vector))

    tate_sum = np.sum(word_vector, axis=0)

    # 使用回数10回以下は省く
    with open("./vectors/word_dict.pkl", "rb") as f:
        word_dict = pickle.load(f)
    with open("./vectors/id_to_word.pkl", "rb") as f:
        id_to_word = pickle.load(f)

    dict_count = 0
    red_word_dict = {}
    red_id_to_word = {}
    for i in range(len(word_dict)):
        if tate_sum[i] < 10:
            continue
        else:
            red_word_dict[id_to_word[i]] = dict_count
            red_id_to_word[dict_count] = id_to_word[i]
            dict_count += 1

    print(len(red_word_dict))

    with open("./vectors/red_word_dict.pkl", "wb") as f:
        pickle.dump(red_word_dict, f)
    with open("./vectors/red_id_to_word.pkl", "wb") as f:
        pickle.dump(red_id_to_word, f)

    np.savetxt("./sum_of_tate.csv", tate_sum, delimiter=",")

def calc_svd_gpu(k_num):
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cx

    word_vector = cp.load("./vectors/word_vector.npy")
    print(word_vector.shape)
    Y, S, Wt = cx.svds(word_vector, k=k_num)

    np.save("./vectors/D.npy", Y)
    np.save("./vectors/S.npy", S)
    np.save("./vectors/V.npy", Wt)

    dot = cp.dot(Wt, Wt.T)
    R = cp.dot(cp.linalg.pinv(dot), Wt)

    cp.save("./vectors/R.npy", R) 

def calc_svd_cpu(k_num):
    word_vector = np.load("./vectors/red_word_vector.npy")
    print(word_vector.shape)
    print(sys.getsizeof(word_vector))
    Y, S, Wt = sp.sparse.linalg.svds(word_vector, k=k_num)

    np.save("./vectors/D.npy", Y)
    np.save("./vectors/S.npy", S)
    np.save("./vectors/V.npy", Wt)

    dot = np.dot(Wt, Wt.T)
    R = np.dot(np.linalg.pinv(dot), Wt)

    np.save("./vectors/R.npy", R)    


if __name__ == "__main__":
    print("-------------------------")
    print(" beginning of the script\n")

    # pick_orgs()
    # create_dict()
    # pre_count()
    # pre_calc()
    # calc_svd_cpu(1000)
    # calc_svd_gpu(1000)

    print("\n    end of the script")
    print("-------------------------")
