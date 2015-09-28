#===============================================================================
# Imports
#===============================================================================

import copy


#===============================================================================
# Functions
#===============================================================================

def calc_f1_score(result, gold):
    intersection = result.intersection(gold)
    
    if len(result) == 0:
        precision = 1
    else:
        precision = float(len(intersection))/len(result)
    
    if len(gold) == 0:
        recall = 1
    else:
        recall = float(len(intersection))/len(gold)
    
    if precision == recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall) # = 2/((1/precision)+(1/recall)) but without devision by 0


#===============================================================================
# Classes
#===============================================================================

class ShallowCopyProxy(object):
    
    def __init__(self, referent):
        self.referent = referent
    
    def __deepcopy__(self, memo):
            """
            This wrapper class cannot be deep-copied. It is always shallow-copied.
            """
            return copy.copy(self)  


class NumContainer(object):

    def __init__(self, num=0):
        self.num = num
