import numpy

def preprocess(text : str) :
    text = text.lower()
    text = text.replace("."," .")
    words = text.split()

    id_to_word = {}
    word_to_id = {}
    for w in words :
        if w not in word_to_id :
            new_id = len(id_to_word)
            id_to_word[new_id] = w
            word_to_id[w] = new_id

    corpus = [word_to_id[w] for w in words]
    corpus = numpy.array(corpus)

    return corpus,id_to_word,word_to_id

def create_co_matrix(text : str) :
    corpus, id_to_word, word_to_id = preprocess(text)
    words_quantity = len(word_to_id)
    co_matrix = numpy.zeros((words_quantity,words_quantity), dtype = numpy.int32)

    for i in range(len(corpus)) :
        left_idx = i - 1
        right_idx = i + 1
    
        if left_idx >= 0 :
            co_matrix[corpus[i]][corpus[left_idx]] += 1
    
        if right_idx < len(corpus) :
            co_matrix[corpus[i]][corpus[right_idx]] += 1

    return co_matrix

def PPMI(text : str, eps = 1e-8) :
    corpus, id_to_word,word_to_id = preprocess(text)
    C = create_co_matrix(text)
    S = C.sum(axis = 0)
    ppmi = numpy.zeros_like(C, dtype = numpy.float32)
    for i in range(len(word_to_id)) :
        for j in range(len(word_to_id)) :
            ppmi[i][j] = max(0,numpy.log2( C[i][j] * C.sum() / (S[i] * S[j])  + eps ))

    return ppmi

    

def cos_similarity(x, y, eps = 1e-8) :
    return numpy.dot(x,y) / (numpy.linalg.norm(x)*numpy.linalg.norm(y))

def most_similiar(query : str, text : str, matrix : numpy.array, top = 5) :
    corpus, id_to_word,word_to_id = preprocess(text)

    query_vector = matrix[word_to_id[query]]
    
    length = len(word_to_id)
    similarity = numpy.zeros(length)
    for i in range(length) :
        similarity[i] = cos_similarity(matrix[i], query_vector)

    cnt = 0;
    for i in ((-1 * similarity).argsort()) :
        if (id_to_word[i] == query) :
            continue
        print("{} : {}".format(id_to_word[i], similarity[i]))
        cnt += 1
        if cnt >= top :
            break