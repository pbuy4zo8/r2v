'''
    written by K.Kobayashi(jeffy890)
    2024.07.24

    vector search system
'''


import gzip
import csv
import datetime
import random
import string
import os
import pandas as pd
import numpy as np
import pickle
import MeCab
import unicodedata
import re


from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory

app = Flask(__name__)
mcb = MeCab.Tagger()

def bunkatsu(text):
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
            if len(word) == 1 and re.match('[a-z]', word) != None:
                pass
            elif hinshi == "名詞" and hinshi2 in hinshi_list:
                temp_word = str(word).lower()
                words_list.append(unicodedata.normalize("NFKC", temp_word))
            node_title = node_title.next

        return words_list
    
    except:
        return []

@app.route("/")
def index():
    return render_template("index.html", version=ver_info)

@app.route("/instructions")
def instructions():
    return render_template("instructions.html", version=ver_info, size_K=K, size_V=V)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/version")
def version():
    return ver_info

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static/images"), "favicon.ico")

@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == 'POST':
        src_words = request.form["src_words"].replace("\n", "").replace("\r\n", "")
        bunkatsu_words = bunkatsu(src_words)
        # print(src_words)

        if src_words == "":
            return render_template('search.html', error_msg="> please put some search words <")
        else:
            post_time = datetime.datetime.now()
            try:
                with open("./src_data.csv", "a", encoding="utf_8_sig") as f:
                    f.write(str(post_time) + ',' + str(request.remote_addr) + '\n')
                with open("./log/"+str(post_time), "w") as f:
                    f.write(src_words)
            except:
                print("file open error? (src_data.csv)")
            result = return_list(src_words)

            return render_template(
                "searched.html", 
                src_words=src_words, 
                input_words="search words: "+src_words, 
                bunkatsu_words=bunkatsu_words, 
                result_list=result
            )
        
    else:
        return render_template('search.html')

@app.route("/iplist")
def iplist():
    if request.method == "GET":
        try:
            server_name = request.args.get("server_name")
            if server_name != None:
                get_time = datetime.datetime.now()
                with open("./iplist.csv", "a", encoding="utf_8_sig") as f:
                    f.write(str(get_time) + "," + str(request.remote_addr) + "," + str(server_name) + "\n")
        except:
            pass

        iplist_df = pd.read_csv("./iplist.csv")
        iplist_df = iplist_df.drop_duplicates(keep="last", subset="name")
        return_str = ""
        for index, row in iplist_df.iterrows():
            return_str = return_str + str(row["time"]) + ", " + str(row["ip"]) + ", " + str(row["name"]) + "<br>"

        return return_str

@app.route("/accessreport")
def accessreport():
    if request.method == "GET":
        try:
            access_type = request.args.get("type")
            print(access_type)
            access_date = datetime.datetime.now()
            with open("./accessreport.csv", "a", encoding="utf_8_sig") as f:
                f.write(str(access_date) + "," + str(access_type) + "\n")
        except:
            pass

    return "200"

def return_list(src_words):
    result_df = calc_inner(src_words)
    
    title_list = []
    researcher_list = []
    rsc_id_list = []
    researcher_id_list = []

    for index, row in result_df.iterrows():
        try:
            researcher = row["研究代表者"].split(" ")
            researcher_name = researcher[0] + researcher[1]
            researcher_id = row["研究代表者"].split("(")[1].replace(")", "")
        except:
            researcher_name = "-"
            researcher_id = ""
    
        temp_title = str(row["研究課題名"])
        if len(temp_title) > 80:
            temp_title = temp_title[:80]
        title_list.append(temp_title)

        researcher_list.append(researcher_name)
        researcher_id_list.append(researcher_id)
        rsc_id_list.append(str(row["研究課題/領域番号"]))

    score_list = np.round(result_df["score"].to_numpy(), 2)

    result_list = zip(
        rsc_id_list, 
        score_list, 
        researcher_id_list, 
        researcher_list, 
        title_list
    )
    return result_list

def calc_inner(src_words):
    v = query_calc(src_words)
    # score = np.dot(doc_vec, v)
    # score = cos_sim(doc_vec, v.T)
    score = cos_similarity(doc_vec, v)

    global base_df

    base_df["score"] = score
    return_df = base_df.sort_values("score", ascending=False)[0:20]
    
    return return_df

def cos_sim(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)

    return np.inner(nx, ny)

def cos_similarity(x, y):
    x_mag = np.sqrt(np.sum(x*x, axis=1))
    y_mag = np.sqrt(np.sum(y*y, axis=0))

    D_len = 7096
    y_b = np.broadcast_to(y.T, (D_len, 100))
    
    return np.sum(x*y_b, axis=1)/x_mag

def query_calc(src_words):
    query_array = np.zeros([V, 1], dtype=float)
    test_query_array = bunkatsu(src_words)
    for i in range(len(test_query_array)):
        if test_query_array[i] in word_dict:
            query_array[word_dict[test_query_array[i]]] += 1
    Z = np.sum(query_array)
    pmi = np.log(query_array/Z) - lp
    pmi = np.where(pmi<0, 0, pmi)
    query_vec = np.dot(R, pmi)
    norm_query = query_vec / np.sqrt(np.dot(query_vec.T, query_vec)[0])

    return norm_query

def load_model():
    with gzip.open("./data.gzip", "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    # データの準備
    base_df = pd.read_csv("./data/base.csv")
    with open("./vectors/word_dict.pkl", "rb") as f:
        word_dict = pickle.load(f)

    model = load_model()

    R = model["R"]
    K, V = R.shape
    doc_vec = model["docvec"]
    lp = model["lp"].reshape([V, 1])

    ver_info = "ver. 0.0.3"
    
    app.run(host="0.0.0.0", port=5001, debug=True)
