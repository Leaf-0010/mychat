# 机器问答实践项目

## 项目概述

该程序能够实现简单的机器问答，并对ES中检索到的数据进行评分和排序，返回可信的结果；

基于ES数据库中的知识数据，对一段文本进行简单的逻辑推理；

## 主要程序说明（views.py）

```python
# 其中需要修改ES中索引信息
def checkSimilarQuestion(quesstr):
    text1 = quesstr.find("请判断：")
    if text1 == 0:
        quesstr = quesstr[4:]
        print(quesstr)
        output = checkSimilarAnswer(quesstr)
        if output:
            returnText = "正确" + "(" + "参考答案：" + str(output[0][0]) + ")" + '\n'
        else:
            returnText = "错误" + '\n'
        return [returnText]

    returnText="在问答库中没有找到答案"
    key="question"
    sim = Similarity()
    senttext1=quesstr   #"解释exists查询的作用"
    bd=es_search_body(senttext1, key)
    print(senttext1)
    # 请在此修改ES中索引名称
    results=es.search(body=bd,index='courseqa')
    #results=es.search(body=bd,index='search_new')
    l=len(results['hits']['hits'])
    print(l)
    similar_texts = []
    for i in range(l):
        senttext2 = results['hits']['hits'][i]['_source']['question']
        y = sim.get_score(senttext1, senttext2)
        if y >= 0.82:
            returnText = results['hits']['hits'][i]['_source']['answer'] + "  （来自问答：" + senttext2 + " )"
            similar_texts.append((returnText, y))

    if similar_texts:
        # 根据相似度评分（y）进行排序
        similar_texts.sort(key=lambda x: x[1], reverse=True)
        output_list = []
        for text, score in similar_texts:
            output_list.append(text + " score: " + str(score) + '\n')
    else:
        max_score = 0.0
        max_text = ""
        for i in range(l):
            senttext2 = results['hits']['hits'][i]['_source']['question']
            y = sim.get_score(senttext1, senttext2)
            if y > max_score:
                max_score = y
                max_text = results['hits']['hits'][i]['_source']['answer'] + "  （来自问答：" + senttext2 + " )"
        if max_score > 0.0:
            output_list = [returnText + "(已搜到的最相似的问题：" + max_text + ")" + " score: " + str(max_score) + '\n']
        else:
            output_list = [returnText]
    return output_list

# 再次修改前端源文件，chatindex.html已在文件中给出，可根据需要修改
def index(request):
    #name = "Hello DTL!"
    data = {}
    data['name'] = "Tanch"
    data['message'] = "你好"
    # return render(request,"模板文件路径",context={字典格式:要在客户端中展示的数据})
    # context是个字典
    #return render(request,"./index.html",context={"name":name})
    return render(request,"./chatindex.html",data)
```

## 项目运行

### 环境要求(主要依赖库及版本信息)

`# Name                  Version 
django                    4.2.1 
elastic-transport         8.4.0  
elasticsearch             8.4.2  
jieba                     0.42.1 
keras                     2.6.0  
keras-bert                0.89.0 
matplotlib                3.7.1  
numpy                     1.23.4 
pandas                    2.0.1  
pillow                    9.5.0  
protobuf                  3.20.0 
python                    3.9.16 
pytorch                   2.0.1 
scikit-learn              1.2.2 
scipy                     1.10.1 
tensorflow                2.8.0 
text2vec                  1.1.8 
torchaudio                2.0.2 
torchvision               0.13.1               `

### 运行项目

1、运行ElasticSearch（请修改views.py相关配置，见上文）

2、配置虚拟环境

3、在虚拟环境中进入mychat所在文件,输入`python manage.py runserver`

4、在Web端访问`http://localhost:8080/index`

