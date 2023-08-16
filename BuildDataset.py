import random

noun_file_path='D:/data/noun.txt'
what_file_path='D:/data/what_sentpattern.txt'
verb_file_path='D:/data/verb.txt'
how_file_path='D:/data/how_sentpattern.txt'

noun_set=[]
with open(noun_file_path, 'r', encoding='utf-8') as nounfin:
    for line in nounfin:
        line=line.strip()
        if(len(line.strip())>1):
            noun_set.append(line)
nounfin.close()

what_set=[]
with open(what_file_path, 'r', encoding='utf-8') as whatfin:
    for line in whatfin:
        line=line.strip()
        if(len(line.strip())>1):
            what_set.append(line)
whatfin.close()

verb_set=[]
with open(verb_file_path, 'r', encoding='utf-8') as verbfin:
    for line in verbfin:
        line=line.strip()
        if(len(line)>1):
            verb_set.append(line)
verbfin.close()

how_set=[]
with open(how_file_path, 'r', encoding='utf-8') as howfin:
    for line in howfin:
        line=line.strip()
        if(len(line.strip())>1):
            how_set.append(line)
howfin.close()

pair_posset=[]
wfile_path= 'pair_pos.csv'
with open(wfile_path, 'w', encoding='utf-8') as fout:
    for i in range(len(noun_set)):
        str1 = noun_set[i]
        for j in range(len(what_set)):
            str2=what_set[j]

            str2=str2.replace("A",str1)
            for k in range(len(what_set)):
                if (j!=k):
                    str3=what_set[k]
                    str3=str3.replace("A",str1)
                    pair_posset.append(str2+" , "+str3)
                    fout.write(str2+" , "+str3+"\n")

    for i in range(len(verb_set)):
        str1 = verb_set[i]
        for j in range(len(how_set)):
            str2 = how_set[j]
            str2 = str2.replace("A", str1)
            for k in range(len(how_set)):
                if (j != k):
                    str3 = how_set[k]
                    str3 = str3.replace("A", str1)
                    pair_posset.append(str2 + " , " + str3)
                    fout.write(str2 + " , " + str3 +"\n")


fout.close()


wfile_path= 'pair_neg.csv'
with open(wfile_path, 'w', encoding='utf-8') as fout:
    for i in range(150):
        j1=random.randint(0,len(noun_set)-1)
        j2=random.randint(0,len(noun_set)-1)
        if abs(j1-j2)>=5:
            str1 = noun_set[j1]
            str2 = noun_set[j2]
        for j in range(10):
            k1 = random.randint(0,len(what_set)-1)
            k2 = random.randint(0,len(what_set)-1)
            if (k1!=k2):
                str3=what_set[k1]
                str4=what_set[k2]
                str5=str3.replace("A",str1)
                str6=str3.replace("A", str2)
                str7 = str4.replace("A", str1)
                str8 = str4.replace("A", str2)
                fout.write(str5 + " , " + str6 +"\n")
                fout.write(str6 + " , " + str7 +"\n")
                fout.write(str5 + " , " + str8 +"\n")
                fout.write(str7 + " , " + str8 +"\n")


    for i in range(100):
        j1=random.randint(0,len(verb_set)-1)
        j2=random.randint(0,len(verb_set)-1)
        if abs(j1-j2)>=5:
            str1 = verb_set[j1]
            str2 = verb_set[j2]
        for j in range(10):
            k1 = random.randint(0,len(how_set)-1)
            k2 = random.randint(0,len(how_set)-1)
            if (k1!=k2):
                str3=how_set[k1]
                str4=how_set[k2]
                str5=str3.replace("A",str1)
                str6=str3.replace("A", str2)
                str7 = str4.replace("A", str1)
                str8 = str4.replace("A", str2)
                fout.write(str5+" , "+str6+"\n")
                fout.write(str5 + " , " + str7 +"\n")
                fout.write(str5 + " , " + str8 +"\n")
                fout.write(str6 + " , " + str7 +"\n")
                fout.write(str6 + " , " + str8 +"\n")
                fout.write(str7 + " , " + str8 +"\n")

    for i in range(170):
        j1=random.randint(0,len(verb_set)-1)
        j2=random.randint(0,len(noun_set)-1)
        str1 = verb_set[j1]
        str2 = noun_set[j2]
        for j in range(10):
            k1 = random.randint(0,len(how_set)-1)
            k2 = random.randint(0,len(what_set)-1)
            str3=how_set[k1]
            str4=what_set[k2]
            str5=str3.replace("A",str1)
            str6=str4.replace("A", str2)
            fout.write(str5+" , "+str6+"\n")
            fout.write(str6 + " , " + str5 +"\n")

fout.close()