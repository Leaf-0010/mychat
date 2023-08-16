#! -*- coding:utf-8 -*-

import json
import numpy as np
from tensorflow import keras
import string
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.callbacks import Callback
from tqdm import tqdm
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

maxlen = 200
learning_rate=5e-5
min_learning_rate = 1e-5

config_path = './bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './bert/chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

# readin data

train_path="D:/data/ZhSquad/train-v1.1-zh.json"
eval_path="D:/data/ZhSquad/dev-v1.1-zh.json"

#Preprocess the data
#Go through the JSON file and store every record as a SquadExample object.


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.input_ids=[]
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)[0]
        self.context_token_to_char = tokenized_context
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for i in range(start_char_idx,end_char_idx):
            ans_token_idx.append(i)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        self.start_token_idx = ans_token_idx[0]
        self.end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)[0]


        # Create inputs
        self.input_ids = tokenized_context + tokenized_question[1:]
        self.token_type_ids = [0] * len(tokenized_context) + [1] * len(
            tokenized_question[1:]
        )
        self.attention_mask = [1] * len(self.input_ids)
        if (len(self.input_ids)>maxlen):
            dist = 20
            a = maxlen - len(tokenized_question[1:])
            if (a > self.start_token_idx + dist and a > self.end_token_idx + dist):
                self.input_ids = tokenized_context[:a - 1] + tokenized_question[1:]
                self.token_type_ids = [0] * len(tokenized_context[0:a - 1]) + [1] * len(
                    tokenized_question[1:]
                )
                self.attention_mask = [1] * len(self.input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed

        padding_length = maxlen - len(self.input_ids)
        if padding_length > 0:  # pad
            self.input_ids =self.input_ids + [0] * padding_length
            self.attention_mask = self.attention_mask + [0] * padding_length
            self.token_type_ids = self.token_type_ids + [0] * padding_length
            return
        elif padding_length < 0:  # skip
            self.skip = True
            return




with open(train_path,'r', encoding='utf-8') as f:
    raw_train_data = json.load(f)

with open(eval_path,'r', encoding='utf-8') as f:
    raw_eval_data = json.load(f)

def create_squad_examples(raw_data):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = SquadExample(
                    question, context, start_char_idx, answer_text, all_answers
                )
                squad_eg.preprocess()
                if (squad_eg.skip == False) :
                    squad_examples.append(squad_eg)
    return squad_examples
def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
 #       dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y

train_squad_examples = create_squad_examples(raw_train_data)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = create_squad_examples(raw_eval_data)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

data = []

for i in range(len(train_squad_examples)):
    if (train_squad_examples[i].skip == False):
        text1=train_squad_examples[i].context.strip()
        text2=train_squad_examples[i].question.strip()
        startidx=train_squad_examples[i].start_token_idx
        endidx = train_squad_examples[i].end_token_idx
        answer=train_squad_examples[i].answer_text
        all_answers = train_squad_examples[i].all_answers
        data.append((text1, text2, startidx,endidx,answer, all_answers))


for i in range(len(eval_squad_examples)):
    if (train_squad_examples[i].skip == False):
        text1=eval_squad_examples[i].context.strip()
        text2=eval_squad_examples[i].question.strip()
        startidx=train_squad_examples[i].start_token_idx
        endidx = train_squad_examples[i].end_token_idx
        answer=eval_squad_examples[i].answer_text
        all_answers = train_squad_examples[i].all_answers
        data.append((text1, text2, startidx,endidx,answer, all_answers))


# 按照9:1的比例划分训练集和验证集
random_order = np.arange(len(data))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

additional_chars = set()
for d in train_data + valid_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[4]))

additional_chars.remove(u'，')

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1

# class data_generator:
#     def __init__(self, data, batch_size=16):
#         self.data = data
#         self.batch_size = batch_size
#         self.steps = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1
#     def __len__(self):
#         return self.steps
#     def __iter__(self):
#         while True:
#             idxs = np.arange(len(self.data))
#             np.random.shuffle(idxs)
#             X1, X2, S1, S2 = [], [], [], []
#             for i in idxs:
#                 d = self.data[i]
#                 text1 = d[0][:maxlen]
#                 text2 = d[1][:maxlen]
#                 tokens = tokenizer.encode(first=text1, second=text2)
#                 an = d[4]
#                 an_tokens=tokenizer.encode(an)[1:-1]
#                 s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
#                 start = list_find(tokens, an_tokens)
#                 if start != -1:
#                     end = start + len(an_tokens) - 1
#                     s1[start] = 1
#                     s2[end] = 1
#                     x1, x2 = tokenizer.encode(first=text1, second=text2)
#                     X1.append(x1)
#                     X2.append(x2)
#                     S1.append(s1)
#                     S2.append(s2)
#                     if len(X1) == self.batch_size or i == idxs[-1]:
#                         X1 = seq_padding(X1)
#                         X2 = seq_padding(X2)
#                         S1 = seq_padding(S1)
#                         S2 = seq_padding(S2)
#                         yield [X1, X2, S1, S2], None
#                         X1, X2, S1, S2 = [], [], [], []


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True


## QA Model
input_ids = Input(shape=(None,)) # 待识别句子输入
token_type_ids = Input(shape=(None,)) # 待识别句子输入
attention_mask = Input(shape=(None,)) # 实体左边界（标签）
# embedding = bert_model([input_ids, token_type_ids, attention_mask])
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(input_ids)

embedding = bert_model([input_ids, token_type_ids])

start_logits = Dense(1, name="start_logit", use_bias=False)(embedding)
# start_logits = Flatten()(start_logits)
start_logits = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([start_logits, x_mask])

end_logits = Dense(1, name="end_logit", use_bias=False)(embedding)
#end_logits = Flatten()(end_logits)
end_logits = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([end_logits, x_mask])

start_probs = Activation(keras.activations.softmax)(start_logits)
end_probs = Activation(keras.activations.softmax)(end_logits)

train_model = keras.Model(
    inputs=[input_ids, token_type_ids],
    outputs=[start_probs, end_probs],
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(lr=5e-5)
train_model.compile(optimizer=optimizer, loss=[loss, loss])

train_model.summary()


def common_start(sa, sb):
# returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return
    return ''.join(_iter())


def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class ExactMatch(keras.callbacks.Callback):
    """
    Each `SquadExample` object contains the character level offsets for each token
    in its input paragraph. We use them to get back the span of text corresponding
    to the tokens between our predicted start and end tokens.
    All the ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = train_model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            # adjust the answer scope
            if(start>end):
                 temp=end
                 end=start
                 start=temp-1
            else:
                 end+=1

            if start >= len(offsets):
                continue
            # pred_char_start = offsets[start][0]
            if end < len(offsets):
            #    pred_char_end = offsets[end][1]
            #     pred_ans = squad_eg.context[pred_char_start:pred_char_end]
                pred_ans = squad_eg.context[start:end]
            else:
            #    pred_ans = squad_eg.context[pred_char_start:]
                pred_ans = squad_eg.context[start:]
            if(idx<10):
                print(squad_eg.context)
                print(squad_eg.question)
                print(squad_eg.all_answers)
                print(start)
                print(end)
                print(pred_ans)
            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            # adjust accuracy computation
            if normalized_pred_ans in [_ for _ in normalized_true_ans]:
                count += 1
            else:
                for a in normalized_true_ans:
                    if (a in pred_ans):
                        count+=1
                        break;

            # if normalized_pred_ans in normalized_true_ans:
            #     count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")

# exact_match_callback = ExactMatch(x_eval, y_eval)
# train_model.fit(
#     x_train,
#     y_train,
#     epochs=2,  # For demonstration, 3 epochs are recommended
#     verbose=2,
#     batch_size=16,
#     callbacks=[exact_match_callback],
# )
#
# print("finish model training!")
#
# # 模型保存
# train_model.save('ChineseQAmodel.h5')
# print("Model saved!")

from keras.models import load_model
from keras_bert import get_custom_objects

train_model = load_model("ChineseQAmodel.h5", custom_objects=get_custom_objects())

# 对单句话进行预测
def predict_single_text(text1,text2):
    # 利用BERT进行tokenize
    pred_ans=""
    squad_examples = []
    context = text1
    question = text2
    answer_text = "answer"
    all_answers = ["answers1", "answers1"]
    start_char_idx = 1
    squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
    squad_eg.preprocess()
    if (squad_eg.skip == False):
        squad_examples.append(squad_eg)

    x_test, y_test = create_inputs_targets(squad_examples)
    pred_start, pred_end = train_model.predict(x_test)
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        squad_eg = squad_examples[idx]
        offsets = squad_eg.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        # adjust the answer scope
        if (start > end):
            temp = end
            end = start
            start = temp - 1
        else:
            end += 1

        if start >= len(offsets):
            continue
        # pred_char_start = offsets[start][0]
        if end < len(offsets):
            #    pred_char_end = offsets[end][1]
            #     pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            pred_ans = squad_eg.context[start:end]
        else:
            #    pred_ans = squad_eg.context[pred_char_start:]
            pred_ans = squad_eg.context[start:]
    return pred_ans


# text1="在Transformer中，多头自注意力用于表示输入序列和输出序列，不过解码器必须通过掩蔽机制来保留自回归属性。Transformer的优点是并行性非常好，符合目前的硬件（主要指GPU）环境。"
# text2="Transformer有什么优点？"

# text1="第1步是计算“station”向量与句中其余每个单词之间的相关性分数。这就是注意力分数。我们简单地使用两个词向量的点积来衡量二者的关系强度。"
# text2="什么是注意力分数？"
#
# text1="小敏今天用5元钱买了5个饼"
# text2="小敏买饼的费用是多少？"


# y=predict_single_text(text1,text2)
#
# print(y)

for i in range(10):
    d = data[i+1000]
    text1=d[0][:maxlen]
    text2 = d[1][:maxlen]

    y=predict_single_text(text1,text2)
    print("上下文："+text1)
    print("问题："+ text2)
    print("抽取的答案："+ y)

