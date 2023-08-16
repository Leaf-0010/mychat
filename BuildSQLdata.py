import random

# file_path='D:/data/table.txt'

# file_path='D:/data/valtable.txt'

file_path='table.txt'

table_set = []
id=0
with open(file_path, 'r', encoding='utf-8') as fin:
    for line in fin:
        if (len(line)<10): break
        textlist=line[:-1].split('，')
        sample={}
        text=textlist[0].strip()
        pos = text.find(':')
        sample['table']=text[:pos]
        sample['tableEntity']=text[pos+1:]
        attnum=len(textlist)
        att=[]
        for i in range(attnum-1):
            attelem={}
            text=textlist[i+1].strip()
            pos=text.find(':')
            attelem['zh']=text[:pos]
            temptext=text[pos+1:]
            post = temptext.find('(')
            attelem['en'] = temptext[:post]
            attelem['type'] = temptext[post + 1:-1]
            att.append(attelem)
        sample['attribute']=att
        table_set.append(sample)

fin.close()
i=0


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return [-1]

def textReplace(text, s):
    newtext=text
    if (s>0):  #requirement preprocessing sql
        posv = find_all(sample['tablezh'], newtext)
        pos1= posv[-1]
        if (pos1>=0):
            newtext=newtext[:pos1]+newtext[pos1:].replace(sample['tablezh'],'table')
        latt=len(sample['attribute'])
        for i in range(latt):
            pos1 = newtext.find(sample['attribute'][i]['zh'])
            if (pos1 >= 0):
                newtext = newtext[:pos1]+newtext[pos1:].replace(sample['attribute'][i]['zh'], 'att'+str(i))
        #calculate OP processing
        algOpNum=len(algOpTextZh)
        for i in range(algOpNum):
            posv=find_all(algOpTextZh[i], newtext)
            if (len(posv)==1 and posv[0]<0): continue
            else:
                offset=0
                li=len(algOpTextZh[i])
                for j in range(len(posv)):
                    pos1=posv[j]+offset
                    if (pos1>0 and newtext[pos1-1:pos1].isdigit() and newtext[pos1+li:pos1+li+1].isdigit()):
                        pos1+=li
                        for j in range(5):
                            if (newtext[pos1+j].isdigit()): continue
                            else: break
                        digitValue=newtext[pos1:pos1+j]
                        pos2=newtext[:pos1+j].rfind("att")
                        if (pos2>=0):
                            index=newtext[pos2+3:pos1-len(algOpTextZh[i])]
                            newtext=newtext[:pos1]+newtext[pos1:pos1+j+1].replace(digitValue, "value"+index)+newtext[pos1+j+1:]
                            offset+=len("value"+index)-len(digitValue)
                            sample['attribute'][int(index)]['value']=str(digitValue)

        #category OP processing
        cateOpNum=len(cateOpZh)
        for i in range(cateOpNum):
            pos1=newtext.find(cateOpZh[i])
            cateValue=""
            if (pos1>=0):
                newtext1=newtext[:pos1]
                newtext2=newtext[pos1:]
                # pos1+=len(cateOpZh[i])
                pos2=newtext2.find("'")
                for j in range(1,5):
                    if (newtext2[pos2+1+j]=="'"): break
                cateValue=newtext2[pos2+1:pos2+1+j]
                pos4 = newtext1.rfind("att")
                if (pos4>=0):
                    index=newtext1[pos4+3:]
                    newtext=newtext1+newtext2[pos2-len(cateOpZh[i]):].replace(cateValue, "value"+index)
                    sample['attribute'][int(index)]['value']="'"+cateValue+"'"

    else:   #sql statement preprocessing
        pos1=newtext.find(sample['tableen'])
        if (pos1>=0):
            newtext=newtext.replace(sample['tableen'],'table')
        latt=len(sample['attribute'])
        for i in range(latt):
            pos1 = newtext.find(sample['attribute'][i]['en'])
            if (pos1 >= 0):
                newtext = newtext.replace(sample['attribute'][i]['en'], 'att'+str(i))

        #calculate OP processing
        algOpNum=len(algOpTextEn)
        for i in range(algOpNum):
            posv=find_all(algOpTextEn[i], newtext)
            if (len(posv)==1 and posv[0]<0): continue
            else:
                li = len(algOpTextEn[i])
                for j in range(len(posv)):
                    pos1=posv[j]
                    if (pos1>0 and newtext[pos1-1:pos1].isdigit() and (newtext[pos1+li:pos1+li+1].isdigit() or newtext[pos1+li:pos1+li+1]=="'")):
                        pos1+=li
                        pos2=newtext[:pos1].rfind("att")
                        if (pos2>=0):
                            index=newtext[pos2+3:pos1-len(algOpTextEn[i])]
                            if (sample['attribute'][int(index)]['type']=='int'):
                                digitValue= str(sample['attribute'][int(index)]['value'])
                                newtext=newtext[:pos1]+newtext[pos1:pos1+len(digitValue)+1].replace(digitValue, "value"+index)+newtext[pos1+len(digitValue)+1:]
                            else:
                                cateValue = sample['attribute'][int(index)]['value']
                                newtext = newtext[:pos1]+newtext[pos1:pos1+len(cateValue)+1].replace(cateValue,"'value" + index + "'")+newtext[pos1+len(cateValue)+1:]


    return newtext



def standarizeRequirement(text,sqltext):
    newtext="在表格"
    newsql="select"

    textlist = text.split(',')
    text = textlist[0].strip()
    pos1 = text.find('表格')
    pos2 = text.find('(')
    pos3 = text.find(')')
    sample['tablezh'] = text[pos1+2:pos2]
    sample['tableen'] = text[pos2 + 1: pos3]
    newtexttable='table'+text[pos3+1:]+","
    attnum = len(textlist)
    att = []
    newtextatt=""
    for i in range(1,attnum - 1):   #previous attnum -2
        attelem = {}
        text = textlist[i].strip()
        if(i==1):
            pos1=text.find('属性有')
            text=text[pos1+3:]
            #newtextatt+= text[:pos1+3]
        pos2=0
        for j in range(len(text)):
            if text[j].isascii()==True:
                break
            else:
                pos2+=1
        attelem['zh']=text[:pos2]
        temptext = text[pos2:]
        pos3 = temptext.find('(')
        attelem['en']=temptext[:pos3]
        attelem['type'] = temptext[pos3 + 1:-1]
        attelem['order'] = i
        att.append(attelem)
        newtextatt+="att"+str(i-1)+","
    sample['attribute'] = att

    text = textlist[attnum-1].strip()
    if text.find('sql')>0:
        newlasttext =textReplace(text, 1)
        newtext=newtext+newtexttable+newtextatt+newlasttext
        sqltext =textReplace(sqltext,0)
    else:
        newtext='输入文本不符合规范'
    return newtext, sqltext




# 生成单个表的SQL语句
cmdText=["编写sql语句显示","请写出sql语句列出","写出sql语句选出","编写sql语句读取"]
logicOpTextZh=['并且','或者']
logicOpTextEn=['AND','OR']
# algOpTextEn=['=','<>','>','<','>=','<=','like']
# algOpTextZh=['等于','不等于','大于','小于','大于等于','小于等于','类似于']
algOpTextEn=['=','<>','>=','<=','>','<','like']
algOpTextZh=['等于','不等于','大于等于','小于等于','大于','小于','类似于']
cateOpEn=['=']
cateOpZh=['为']

orderbyTextZh=['升序','降序']
orderbyTextEn=['asc','dec']
orderbyBool=False
orderbyindex = 0
orderbyatt =0


def validate(origfulltext,origsqltext):
    correct=True

    fulltext, sqltext = standarizeRequirement(origfulltext, origsqltext)
    textlist = fulltext.split(',')
    for i in range(len(orderbyTextZh)):
        fpos=fulltext.rfind(orderbyTextZh[i])
        spos=sqltext.rfind(orderbyTextEn[i])
        if (fpos*spos<0):
            correct=False
            print("---------------  error ------------------")
            print(fulltext)
            print(sqltext)
            break

    posv=find_all("att",sqltext)
    lp=len(posv)
    lt=len(textlist)
    for i in range(lp):
        pos=posv[i]
        if (pos<0): break
        sqlatt=sqltext[pos:pos+4]
        find=False
        for j in range(1,lt-1):
            text = textlist[j].strip()
            if (text.find(sqlatt)>=0):
                find=True
                break
        if find==False:
            correct=False
            print("---------------  error ------------------")
            print(fulltext)
            print(sqltext)
            break

    return correct

#在表格students中，包含的属性有学号sid(int), 姓名sname(varchar)，年龄sage(int), 性别 ssex(char)，程序设计成绩sprog(int), 软件工程成绩ssoft(int), 编写sql语句统计所有年龄大于20岁的学生
require_set=[]
sql_set=[]
tnum=len(table_set)
for i in range(tnum):
    # intatt = []
    # charatt = []
    condtext = ""
    optext = ""
    require_text=f"在表格{table_set[i]['tableEntity']}({table_set[i]['table']})中,包含的属性有"
    att=[]
    att=table_set[i]['attribute']
    attnum=len(att)
    if attnum<1: continue

    temp=""
    for j in range(len(att)):
        temp+=att[j]['zh']+att[j]['en']+"("+att[j]['type']+"), "
        # if att[j]['type']=='int':
        #     intatt.append(att[j])
        # if att[j]['type']=='char':
        #     charatt.append(att[j])
    cmdtemp=cmdText[random.randrange(len(cmdText))]
    cmdtemp1=cmdtemp


    require_text+=temp
    require_text_backup=require_text
    #写出一个条件的SQL
    # intnum=len(intatt)-1
    # charnum=len(charatt)
    # onenum=(intnum+charnum)/2
    for k in range(len(att)):
        condtext=att[k]['zh']
        type=att[k]['type']
        if (type=='int'):
            for op in range(6):
                strvalue=str(random.randrange(1,20))
                optext=algOpTextZh[op]+strvalue+"的"+table_set[i]['tableEntity']

                sqltext="select all from "+ table_set[i]['table']+ ' where '+ att[k]['en'] + algOpTextEn[op] + strvalue
                orderbyBool = False
                a = random.randrange(1, 20)
                cmdtemp = cmdtemp1
                if (a >= 10):
                    orderbyBool = True
                    orderbyatt = random.randrange(0, attnum - 1)
                    if (a >= 15):
                        orderbyindex = 1
                        cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]
                    else:
                        orderbyindex = 0
                        cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]
                condtext = att[k]['zh']
                fulltext=require_text+cmdtemp+condtext+optext

                sqltext = "select all from " + table_set[i]['table'] + ' where ' + att[k]['en'] + algOpTextEn[op] + strvalue
                if orderbyBool:
                    sqltext+=" orderby "+att[orderbyatt]['en']+ " "+orderbyTextEn[orderbyindex]
                if (validate(fulltext, sqltext)==True):
                    require_set.append(fulltext)
                    sql_set.append(sqltext)
                else:
                    print("error")
                condtext=""
                optext=""

    for k in range(len(att)):
        condtext = att[k]['zh']
        type = att[k]['type']
        if (type=='char'):
            for op in range(1):
                strvalue=chr(random.randrange(1,20)+ord('A'))
                optext=cateOpZh[op]+"'"+strvalue+"'"+"的"+table_set[i]['tableEntity']
                orderbyBool = False
                a = random.randrange(1, 20)
                cmdtemp = cmdtemp1
                if (a >= 10):
                    orderbyBool = True
                    orderbyatt = random.randrange(0, attnum - 1)
                    if (a >= 15):
                        orderbyindex = 1
                        cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]
                    else:
                        orderbyindex = 0
                        cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]

                fulltext=require_text+cmdtemp+condtext+optext
                sqltext="select all from "+ table_set[i]['table']+ ' where '+ att[k]['en'] + cateOpEn[op] + "'"+strvalue+"'"

                if orderbyBool:
                    sqltext += " orderby " + att[orderbyatt]['en'] + " " + orderbyTextEn[orderbyindex]
                if (validate(fulltext, sqltext) == True):
                    require_set.append(fulltext)
                    sql_set.append(sqltext)
                else:
                    print("error")

                condtext = ""
                optext = ""


    #显示多个字段的信息
    l=len(table_set[i]['attribute'])
    for i1 in range((l-1)*(l-1)):
        if (l<=3): break
        attlist = []
        for i1 in range(l-1):
            a=random.randrange(1,20)
            if(a>10):
                attlist.append(random.randrange(1,l-2))
        attset=set(attlist)   #去重
        attlist=list(attset)
        condtext=""
        optext=""

        for k in range(len(att)):
            if (len(attlist)<1): break
            condtext=att[k]['zh']

            orderbyBool = False
            cmdtemp = cmdtemp1
            a = random.randrange(1, 20)
            if (a >= 10):
                cmdtemp=cmdtemp1
                orderbyBool = True
                orderbyatt = random.randrange(0, attnum - 1)
                if (a >= 15):
                    orderbyindex = 1
                    cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]
                else:
                    orderbyindex = 0
                    cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]

            type = att[k]['type']
            if (type == 'int'):
                for op in range(6):
                    strvalue=str(random.randrange(1,20))
                    optext=algOpTextZh[op]+strvalue+"的"+table_set[i]['tableEntity']

                    attinfo="的"
                    l1=len(attlist)
                    fieldinfo="("
                    for p in range(l1-1):
                        attinfo+=att[attlist[p]]['zh']+"、"
                        fieldinfo+=att[attlist[p]]['en']+","
                    attinfo+=att[attlist[l1-1]]['zh']+"信息"
                    fieldinfo+=att[attlist[l1-1]]['en']+")"
                    condtext = att[k]['zh']
                    fulltext=require_text+cmdtemp+condtext+optext+attinfo
                    sqltext="select "+fieldinfo+" from "+ table_set[i]['table']+ ' where '+ att[k]['en'] + algOpTextEn[op] + strvalue
                    if orderbyBool:
                        sqltext += " orderby " + att[orderbyatt]['en'] + " " + orderbyTextEn[orderbyindex]
                    if (validate(fulltext, sqltext) == True):
                        require_set.append(fulltext)
                        sql_set.append(sqltext)
                    else:
                        print("error")

                    condtext=""
                    optext=""


    cmdtemp=cmdText[random.randrange(len(cmdText))]
    cmdtemp1=cmdtemp

    require_text_backup=require_text
    condtext = ""
    optext = ""


    #写出两个条件的SQL

    attnum=len(att)
    for m in range((attnum-1)*8):
        if (attnum<3): break

        orderbyBool = False
        cmdtemp = cmdtemp1
        a = random.randrange(1, 20)
        if (a >= 10):
            cmdtemp = cmdtemp1
            orderbyBool = True
            orderbyatt = random.randrange(0, attnum - 1)
            if (a >= 15):
                orderbyindex = 1
                cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]
            else:
                orderbyindex = 0
                cmdtemp = cmdtemp[:-2] + "按照" + att[orderbyatt]['zh'] + orderbyTextZh[orderbyindex] + cmdtemp[-2:]

        cond=[]

        for boolop in range(2):
            optext=[]
            opsql=[]
            condtext = []
            op = []
            strvalue = []
            for m1 in range(2):
                cond.append(random.randrange(0,attnum-1))
                condtext.append(att[cond[m1]]['zh'])
                if att[cond[m1]]['type']=='int':
                    tempint=random.randrange(0,5)
                    op.append(tempint)
                    strvalue.append(str(random.randrange(1,20)))
                    optext.append(str(algOpTextZh[tempint] + strvalue[m1]))
                    opsql.append(att[cond[m1]]['en']+algOpTextEn[tempint]+strvalue[m1])
                if att[cond[m1]]['type']=='char':
                    op.append(0)
                    strvalue.append(chr(random.randrange(1,20)+ord('A')))
                    optext.append("为" + "'" + strvalue[m1] + "'")
                    opsql.append(att[cond[m1]]['en'] + algOpTextEn[0] + "'" + strvalue[m1] +"'")

            boolcmbtext=condtext[0]+optext[0]+logicOpTextZh[boolop]+condtext[1]+optext[1]+"的"+table_set[i]['tableEntity']


            fulltext=require_text+cmdtemp+boolcmbtext
            sqltext="select all from "+ table_set[i]['table']+ ' where ('+ opsql[0] +" "+logicOpTextEn[boolop] + " "+ opsql[1]+ ')'
            if orderbyBool:
                sqltext+=" orderby "+att[orderbyatt]['en']+ " "+orderbyTextEn[orderbyindex]

            if (validate(fulltext, sqltext) == True):
                require_set.append(fulltext)
                sql_set.append(sqltext)
            else:
                print("error")

            boolcmbtext = ""




print(len(require_set))

# requirefile_path = 'D:/data/SQLRequirement.txt'
# sqlfile_path = 'D:/data/SQLStatement.txt'

requirefile_path = 'D:/data/valSQLRequirement.txt'
sqlfile_path = 'D:/data/valSQLStatement.txt'

with open(requirefile_path, 'w', encoding='utf-8') as fout1:
    for i in range(len(require_set)):
        fout1.write(require_set[i] + "\n")
    fout1.close()

with open(sqlfile_path, 'w', encoding='utf-8') as fout1:
    for i in range(len(sql_set)):
        fout1.write(sql_set[i] + "\n")
    fout1.close()


