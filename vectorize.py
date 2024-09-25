'''
    written by K.Kobayashi(jeffy890)
    2024.09.24

    ベクトル化スクリプト
'''


import mochihashi as mh  # 持橋先生のスクリプト群

import pickle
import sys

# calc vectors
def vectorization(path, dim=100):
    if path == None:
        print("enter some files path")

    try:
        with open(path, "rb") as f:
            docs = pickle.load(f)

        lp = mh.docvec.unigram(docs)
        print(" done calc lp...")

        matrix = mh.docvec.parse(docs, lp)
        print(" done calc matrix...")

        doc_vec, featvec = mh.docvec.compress(matrix, dim)
        print(" done calc vectors...")

        R = solve(np.dot(featvec.T, featvec), featvec.T)
        print(" done calc R...")

        mh.docvec.save({ 'docvec': doc_vec, 'R' : R, 'lp' : lp }, "./data.gzip")

    except:
        print("seems like there's something wrong in the file")
        print("check file or file path\n")
        print("script ends with sys.exit()\n")
        sys.exit()


if __name__ == "__main__":
    vectorization("./hoge")