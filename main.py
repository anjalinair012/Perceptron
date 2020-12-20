import numpy as np
import matplotlib.pyplot as plt
import math

def create_dataset(P,N):
  Mu = np.random.normal(size=(P,N))
  S = np.random.randint(2,size=(P,1))
  S = np.where(S == 0, -1, 1)
  return Mu, S

def model(Mu, S, maxepochs):
  epoch=0
  update=1
  P = Mu.shape[0]
  N = Mu.shape[1]
  x = np.zeros((N,1))
  y=0
  w = np.zeros(N)

  while update != 0 and epoch < maxepochs:
    epoch = epoch+1
    update=0
    for i in range(P):
      x = Mu[i]
      y = S[i]
      E = np.dot(w,x)*y
      #print(E)
      if E <= 0.5:
        update = update + 1
        #print(w)
        w = w + (1/N)*x*y
        #print(w)
  return w

def compute_P(N,alpha):
  P = []
  for i in range(len(N)):
    P.append(N[i]*np.array(alpha,dtype='int'))
  return np.array(P)

def accuracy(w,Mu,S):
  acc = 0
  for i in range(Mu.shape[0]):
    if np.sign(np.dot(w,Mu[i])) == np.sign(S[i]):
      acc += 1
  return int(acc/Mu.shape[0])

def plotPls(all_acc,alpha):
    plt.figure(figsize=(10, 6))
    plt.style.use('classic')
    x = alpha
    for y in all_acc:
        plt.plot(x, y, marker='o')
        # plt.plot(x,y,'*')
    plt.legend(['N=20', ])
    plt.xlabel('\u03B1 = P/N')
    plt.ylabel('Probability Q(l.s)')
    axes = plt.axes()
    axes.set_ylim([0, 1.2])
    plt.show()

def singlePls(all_acc,all_pls):
    plt.figure(figsize=(10, 6))
    plt.style.use('classic')
    x = alpha
    for y in all_acc:
        plt.plot(x, y, marker='o')
        #plt.plot(x,y,'*')
    plt.plot(x,all_pls, marker='*')
    plt.legend(['$Q_{ls}$','$P_{ls}$'])
    plt.xlabel('\u03B1 = P/N')
    plt.ylabel('ls ratio')
    axes = plt.axes()
    axes.set_ylim([0, 1.2])
    plt.show()

def calcPls(n):
    all_pls=[]
    for a in alpha:
        Pls=0
        p=a*n
        if p<=n:
            Pls=1
        else:
            for i in range(0,n):
                Pls=Pls+math.factorial(p-1)/(math.factorial(i)*math.factorial(p-1-i))
            Pls=Pls*math.pow(2,1-p)
        all_pls.append(Pls)
    return all_pls



nmax=50
N = [20]
alpha = [0.75,1.0,1.25,1.5,1.75,2.0,2.5,2.75,3]
maxepochs=100
all_acc = []
P = compute_P(N, alpha)
print(P)
for n in N:
  acc_alpha = []
  for a in alpha:
    p = int(n*a)
    #print('N={}, P={}, alpha={}'.format(n,p,p/n))
    c = 0
    for d in range(nmax):
      Mu, S = create_dataset(p, n)
      plt.hist(Mu)
      plt.show()
      exit()
      w = model(Mu,S,maxepochs)
      acc = accuracy(w,Mu,S)
      if acc == 1:
        c += 1
    acc_alpha.append(c/nmax)
  all_acc.append(acc_alpha)
  all_pls=calcPls(n)
  plotPls(all_acc,alpha)
  singlePls(all_acc,all_pls)

