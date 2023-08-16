import numpy as np
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX

model_path = './bert/chinese_L-12_H-768_A-12'
texts = ['什么是宇宙', '宇宙是什么','宇宙的概念？', '解释宇宙的定义','什么是诗歌']

def cos(vec1,vec2):
    return np.dot(vec1,vec2) / (np.linalg.norm(vec1) *np.linalg.norm(vec2))

query_vec = extract_embeddings(model_path, texts)

for i in range(1,5):
    print(texts[0]+"  ----  "+ texts[i] +   "  cos_similarity:  "+ str(cos(query_vec[0][0,:],query_vec[i][0,:])))



