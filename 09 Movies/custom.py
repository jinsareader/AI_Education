import re
import numpy

def text_preprocess(text : str) :
    text = re.sub(r"[^0-9a-zA-Z가-힣]",repl=" ",string=text.lower())
    text = re.sub(r"[0-9]+",repl="N",string=text)
    return text
    
def make_dict(sentences : list, word_dict : dict = None) :
    data = " ".join(sentences)
    data = text_preprocess(data).split()
    if word_dict is None :
        word_dict = {}
        word_dict["<pad>"] = 0 #패딩
        word_dict["<unk>"] = 1 #없는 단어
    for w in data :
        if w not in word_dict :
            word_dict[w] = len(word_dict)
    number_dict = {i : w for w, i in word_dict.items()}
    return word_dict, number_dict

def word_num_encoding(sentences : list, word_dict : dict, unk : str = "<unk>") :
    word_size = len(word_dict)
    corpus = []
    max_len = 0
    for s in sentences :
        s = text_preprocess(s).split()
        max_len = max(max_len, len(s))
    for s in sentences :
        s_array = []
        s = text_preprocess(s).split()
        for i in range(max_len) :
            if len(s) <= i :
                s_array.append(0)
                continue
            try :
                s_array.append(word_dict[s[i]])
            except :
                s_array.append(word_dict[unk])
        corpus.append(s_array)
    corpus = numpy.array(corpus)    
    return corpus

def make_comatrix(corpus, word_size, window_size = 1) :
    comatrix = numpy.zeros(shape = (word_size, word_size))
    for s in corpus :
        for w in range(len(s)) :
            if s[w] <= 0 :
                break
            for i in range(1,window_size+1) :
                if w-i >= 0 :
                    comatrix[s[w], s[w-i]] += 1
                if w+i < len(s) :
                    if s[w+i] > 0  :
                        comatrix[s[w], s[w+i]] += 1
    return comatrix

def cos_similarity(x, y) :
    eps = 1e-15
    return numpy.dot(x,y) / (numpy.linalg.norm(x)*numpy.linalg.norm(y) + eps)

def most_similiar(query, word_dict, number_dict, comatrix, top = 5) :
    if query not in word_dict :
        print("{}(이)가 사전에 존재하지 않습니다.".format(query))
        return

    word_size = len(word_dict)
    similiar = numpy.zeros(shape = (word_size))
    for i in range(word_size) :
        similiar[i] = cos_similarity(comatrix[word_dict[query]], comatrix[i])

    print("검색어 ||",query)
    cnt = 0
    for i in (-1 * similiar).argsort() :
        if number_dict[i] == query :
            continue
        print("{} : {}".format(number_dict[i], similiar[i]))
        cnt += 1
        if cnt >= top :
            break
    print("")

def make_pmi(comatrix, verdose = False) :
    P = numpy.zeros_like(comatrix)
    N = numpy.sum(comatrix)
    S = numpy.sum(comatrix, axis = 0)
    eps = 1e-15
    
    cnt = 0
    total = P.shape[0] * P.shape[1]
    
    for i in range(P.shape[0]) :
        for j in range(P.shape[1]) :
            pmi = numpy.log2(comatrix[i, j] * N / (S[i]*S[j] + eps) + eps)
            P[i,j] = max(0, pmi)

            if verdose :
                cnt += 1
                if cnt % (total // 100 + 1)== 0 :
                    print("%.1f%% 완료" %(100*cnt/total))
    
    return P   

def make_word_pair(corpus, window_size = 1) :
    word_pair = []
    for s in corpus :
        for w in range(len(s)) :
            for i in range(1,window_size+1) :
                if w-i >= 0 :
                    temp = [s[w], s[w-i]]
                    word_pair.append(temp)
                if w+i < len(s) :
                    if s[w+i] > 0 :
                        temp = [s[w], s[w+i]]
                        word_pair.append(temp)
    
    return word_pair