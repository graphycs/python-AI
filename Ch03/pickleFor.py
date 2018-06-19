#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pickle.py
#  
#  Copyright 2018 419644 <419644@SF00419644LA>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from math import log
import operator
import pickle


class Person:  
    def __init__(self,n,a):  
        self.name=n  
        self.age=a  
        
    def show(self):  
        print(self.name+"_"+str(self.age)  )
     
aa = Person("JGood", 2)  
aa.show()  
f=open('d:\\p.txt','wb')  
pickle.dump(aa,f)  
f.close()  
#del Person  
f=open('d:\\p.txt','rb')  
bb=pickle.load(f)  
f.close()  
bb.show()  
