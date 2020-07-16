# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:50:49 2020

@author: hlj0812.duapp.com
"""

from matplotlib import pyplot as plt
import numpy as np
files=open('train.txt','r')
readlines=files.readlines()
files.close()
content=[]
i=0
for c in readlines:
    if i%100==0:
        content.append(list(map(float,c.strip('\n').split(' ')[1:])))
    i+=1
content=np.array(content)
x=list(range(1,content.shape[0]+1))
loss1,=plt.plot(x,content[:,0],color='blue')
loss2,=plt.plot(x,content[:,1],color='red')
plt.legend(handles = [loss1,loss2], 
           labels = ['person1','person2'],
           loc='upper right')
plt.title('Loss of Person1 and Person2 for DeekFakes')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.show()
