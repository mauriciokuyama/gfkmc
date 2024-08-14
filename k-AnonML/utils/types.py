# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum


class Dataset(Enum):
    CMC = 'cmc'
    MGM = 'mgm'
    ADULT = 'adult'
    ADULT_CAT = 'adult_cat'
    CAHOUSING = 'cahousing'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class AnonMethod(Enum):
    OLA = 'ola'
    MONDRIAN = 'mondrian'
    TDG = 'tdg'
    CB = 'cb'
    GFKMC = 'gfkmc'
    ORIGINAL = 'original'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


class Classifier(Enum):
    RF = 'rf'
    KNN = 'knn'
    SVM = 'svm'
    XGB = 'xgb'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(other) == self.value


MLRes = namedtuple("MLRes", ['acc', 'precision', 'recall', 'f1_score'])
# python3.6
MLRes.__new__.__defaults__ = (['accuracy'], ['precision'], ['recall'], ['f1'])
