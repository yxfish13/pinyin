import regex
from pypinyin import lazy_pinyin
import json

import codecs
from collections import defaultdict
from collections import Counter
from itertools import chain
import pickle


def getTrainPyHz(sent):
    text = regex.sub(u"[^ \p{Han}]", "", sent).replace(" ","")
    pnyns = lazy_pinyin(text)
    assert len(text)==len(pnyns),"not equal"
    return pnyns,text
trainSet=[]
for fileNameInt in range(1,12):
    with open("sina_news/2016-"+format(fileNameInt,"02d")+".txt") as f:
        cnt = 0
        while 1:
            line = f.readline()
            if not line:break
            data = json.loads(line)
            try:
                trainSet.append(getTrainPyHz(data["html"]))
                cnt = cnt + 1
            except:
                continue
            if (cnt%100==0):print(cnt)
            if (cnt%5000==0):print(data)

                
pickle.dump(trainSet,open("trainSet","wb"))


int2han={}
han2int={}
han2P=defaultdict(int)
with codecs.open("一二级汉字表.txt", "r",encoding='GBK', errors='ignore') as f:
    line = f.readline().encode("UTF-8").decode("UTF-8")
    for num,hanzi in enumerate(line):
        int2han[num]=hanzi
        han2int[hanzi]=num

py2hanzi={}
with codecs.open("拼音汉字表.txt", "r",encoding='GBK', errors='ignore') as f:
    while 1:
            line = f.readline()
            if not line:break
            line = line.encode("UTF-8").decode("UTF-8")
            pyhanzi = line.split();
            py2hanzi[pyhanzi[0]]=pyhanzi[1:]

test = [trainSet[x][1] for x in range(len(trainSet))]
totalNum = 0
for x in range(len(trainSet)):
    totalNum = totalNum + len(trainSet[x][1])

hanzi2cnt = Counter(chain.from_iterable(test))
for hanzi, cnt in hanzi2cnt.items():
    if hanzi in han2int:
        han2P[hanzi]=1.0*cnt/totalNum


pickle.dump(py2hanzi,open("py2hanzi","wb"))
pickle.dump(han2P,open("han2P","wb"))

class Ngram:
    def __init__(self,n):
        from collections import defaultdict
        self.gram =defaultdict(float)
        self.n = n
    def train(self,trainSet):
        for n in range(1,self.n+1):
            print(n,len(trainSet))
            idx = 0
            for sentence in trainSet:
                idx = idx + 1
                if(idx%10000==0):
                    print(idx,len(trainSet))
                ls = len(sentence[1])
                for x in range(n,ls+1):
                    #print(ttt)
                    self.gram[sentence[1][x-n:x]] += 1
ngram = Ngram(4)
ngram.train(trainSet)
pickle.dump(ngram,open("ngram5","wb"))

from collections import defaultdict
ngramdict=defaultdict(float)
for i in range(1,4):
    for x in ngram.gram[i]:
        if (ngram.gram[i][x]>3):
            ngramdict[x]=ngram.gram[i][x]
print(len(ngramdict),len(ngram.gram[3])+len(ngram.gram[2]))

pickle.dump(ngramdict,open("gramdict","wb"))