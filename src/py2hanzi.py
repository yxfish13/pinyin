

import pickle
print("loading models....\n")
py2hanzi = pickle.load(open("model/py2hanzi","rb"))
han2P = pickle.load(open("model/han2P","rb"))
#ngram = pickle.load(open("model/ngram","rb"))
ngram = pickle.load(open("model/gramdict","rb"))
print("already loaded\n")
def translate(inputFile,model):
    ans = []
    try:
        with open(inputFile) as fin:
            while 1:
                line = fin.readline()
                if not line:break
                pys = line.split()
                mypys = model.predict(pys)
                ans.append(mypys)
    except:
        with codecs.open(inputFile, "r",encoding='GBK', errors='ignore') as fin:
            while 1:
                line = fin.readline()
                if not line:break
                pys = line.split()
                mypys = model.predict(pys)
                ans.append(mypys)
    return ans
    
def measure(inputFile,outputFile,model):
    import codecs
    outTxt =[]
    try:
        with open(outputFile) as fout:
            while 1:
                line = fout.readline()
                if not line : break
                outTxt.append(line)
    except:
        with codecs.open(outputFile, "r",encoding='GBK', errors='ignore') as fout:
            while 1:
                line = fout.readline()
                if not line : break
                outTxt.append(line)
    idx = 0
    right = 0
    total = 0
    try:
        with open(inputFile) as fin:
            while 1:
                line = fin.readline()
                if not line:break
                pys = line.split()
                mypys = model.predict(pys)
                print(mypys,outTxt[idx])
                temp = (mypys == outTxt[idx])
                right += sum(temp) 
                total += len(temp)
                idx += 1
                if (idx>50):break
    except:
        with codecs.open(inputFile, "r",encoding='GBK', errors='ignore') as fin:
            while 1:
                line = fin.readline()
                if not line:break
                pys = line.split()
                mypys = model.predict(pys)
                print(mypys,outTxt[idx])
                temp = [1 if mypys[x]==outTxt[idx][x] else 0 for x in range(len(mypys)) ]
                right += sum(temp) 
                total += len(temp)
                idx += 1
                if (idx>50):break
    return right,total,1.0*right/total

class Viterbi:
    def __init__(self,n,ngram,han2P,py2hanzi,alpha,choice = 0):
        self.ngram = ngram
        self.n = n
        self.han2P = han2P
        self.py2hanzi = py2hanzi
        self.alpha = alpha
        self.total={}
        self.choice=choice
    def GetP(self,sentence,py):
        if (self.choice==1) : return 1e-8+self.alpha*self.ngram[sentence[-self.n:]]/(self.ngram[sentence[-self.n:-1]]+(1e-8))+(1-self.alpha)*self.han2P[sentence[-1]]
        count = 0
        Pdict={}
        if not (sentence[:-1],py) in self.total:
            for x in self.py2hanzi[py]:
                count = count + self.ngram[sentence[:-1]+x]
            self.total[(sentence[:-1],py)] = count
        return 1e-8+self.alpha*self.ngram[sentence[-self.n:]]/(self.total[(sentence[:-1],py)]+(1e-8))+(1-self.alpha)*self.han2P[sentence[-1]]        
    def GetallP(self,sentence,py):
        count = 0
        Pdict={}
        for x in self.py2hanzi[py]:
            count = count + self.ngram.gram[len(sentence)+1][sentence+x]
        for x in self.py2hanzi[py]:
            Pdict[x] = 1e-8 + self.alpha * self.ngram.gram[len(sentence)+1][sentence+x]/(1e-6+count)
        return Pdict
    def predict(self,pinyin):
        from collections import defaultdict
        from math import log
        oldopt = defaultdict(float)
        oldstate=[""]
        for i in range(self.n-1):
            newstate=[]
            newopt = defaultdict(float)
            for x in oldstate:
                for y in self.py2hanzi[pinyin[i]]:
                    newstate.append(x+y)
                    newopt[x+y] = oldopt[x]+log(self.GetP(x+y,pinyin[i]))
            oldstate = newstate
            oldopt = newopt
        trace={}
        for i in range(self.n-1,len(pinyin)):
            #print(oldopt.keys())
            newopt={}
            values=[]
            for w in oldopt:
                values.append(oldopt[w])
            values.sort(reverse=True)
            #print(len(values))
            minValue = -1E8
            if (len(values)>1000):
                minValue=values[1000]
            else: 
                minValue = min(values)
            #print(len(values))
            for w in oldopt:
                if oldopt[w]>minValue-1e-7:
                    #prb = self.GetallP(w,pinyin[i])
                    for backup in self.py2hanzi[pinyin[i]]:
                        Pt = oldopt[w]+log(self.GetP(w+backup,pinyin[i]))
                        nw = w[1:]+backup
                        #print(nw,w,backup)
                        if (not nw in newopt)or newopt[nw]<Pt:
                            newopt[nw] = Pt
                            trace[(i,nw)]=(i-1,w)
                    #print(len(oldopt))
            oldopt = newopt
        maxP = -1E10
        maxstate=""
        for x in oldopt:
            #print(x)
            if (oldopt[x]>maxP):
                maxP = oldopt[x]
                maxstate = x
        return self.out(trace,maxstate,len(pinyin)-1,self.n)
    def out(self,trace,state,l,n):
        #print(state,l)
        if (l==n-2):
            return state
        else:
            return self.out(trace,trace[(l,state)][1],trace[(l,state)][0],n)+state[-1]

        
def outResult(inputFile,outputFile,model):
    result = translate(inputFile,model)
    with open(outputFile,"w") as f:
        f.write("\n".join(result))

import sys
trans = Viterbi(3,ngram,han2P,py2hanzi,0.5)
inputFile = sys.argv[1]
outputFile = sys.argv[2]


outResult(inputFile,outputFile,trans)