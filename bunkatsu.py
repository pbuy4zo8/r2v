'''
    written by K.Kobayashi(jeffy890, pbuy4zo8)
    2024.09.25

    text split functions
    テキストを分かち書きする関数群

'''

import MeCab
mcb = MeCab.Tagger("-Owakati")

import sys
import unicodedata
import re

def bunkatsu_no_hinshi(text, flug=0):
    try:
        stand_text = unicodedata.normalize("NFKC", text)
        re_text = re.sub('[0-9]', '', stand_text)
        text_wakati = mcb.parse(re_text).split(" ")
        words_list = []

        for i in range(len(text_wakati)):
            if text_wakati[i] == "\n":
                continue

            temp_word = str(text_wakati[i]).lower()
            words_list.append(unicodedata.normalize("NFKC", temp_word))
            
        return words_list
    
    except:
        return []

def bunkatsu_hinshi(text, flug=0):
    # 入力を分割して名詞のみをリスト化して返す
    hinshi_list = ["一般", "サ変接続", "固有名詞", "接尾", "特殊"]
    
    try:
        stand_text = unicodedata.normalize("NFKC", text)
        re_text = re.sub('[0-9]', '', stand_text)
        node_title = mcb.parseToNode(re_text)
        words_list = []
        while node_title:
            word = node_title.surface
            hinshi = node_title.feature.split(",")[0]
            hinshi2 = node_title.feature.split(",")[1]
            if flug == 1:
                print(word, hinshi, hinshi2)
            if len(word) == 1 and re.match('[a-z]', word) != None:
                pass
            elif hinshi == "名詞" and hinshi2 in hinshi_list:
                temp_word = str(word).lower()
                words_list.append(unicodedata.normalize("NFKC", temp_word))
            node_title = node_title.next

        return words_list
    
    except:
        return []

if __name__ == "__main__":
    print("text bunkatsu function")
    text = input("put some text: ")
    print(text)
    print(bunkatsu(text, flug=1))