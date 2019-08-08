# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:54:06 2019

@author: amcg8
"""

from sklearn import tree
features = [[125,0],[130,0],[150,1], [145,1]]
labels =  [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf=clf.fit(features, labels)
print(clf.predict([[600, 0]]))