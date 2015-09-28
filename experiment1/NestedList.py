#===============================================================================
# Imports
#===============================================================================

import random

from numpy import mean

from utils import NumContainer, calc_f1_score
import settings


#===============================================================================
# Functions
#===============================================================================

def generate_corpus(size, **kwargs):
    corpus = set()
    
    while len(corpus) < size:
        corpus.add(NestedList.generate_random_instance(**kwargs))
    
    return corpus


#===============================================================================
# Classes
#===============================================================================

class NestedListManager(object):

    class ACTIONS(object):
        ADD_CHILD_LEAF = 0
        ADD_CHILD_NODE = 1
        RETURN_TO_PARENT = 2
    
    def __init__(self):
        self.root = NestedList()
        self.current_node = self.root
    
    def add_child_leaf(self, token):
        self.current_node.append(token)
        
    def add_child_node(self, token):
        new_node = NestedList(parent=self.current_node)
        self.current_node.append(new_node)
        self.current_node = new_node
    
    def return_to_parent(self, token):
        # fail silently when the network performs  illegal data structure actions
        if self.current_node.parent != None:
            self.current_node = self.current_node.parent
    
    def do_action(self, action, token):
        
        func_dict = {self.ACTIONS.ADD_CHILD_LEAF: self.add_child_leaf,
                     self.ACTIONS.ADD_CHILD_NODE: self.add_child_node,
                     self.ACTIONS.RETURN_TO_PARENT: self.return_to_parent}
        
        func_dict[action](token)    


class NestedList(list):
    
    def __init__(self, *args, **kwargs):
        parent = kwargs.pop("parent", None)
        super(NestedList, self).__init__(*args, **kwargs)
        self.parent = parent
    
    @classmethod
    def generate_random_instance(cls, 
                                 p_continue=0.6, 
                                 p_new_list=0.3, 
                                 min_num=settings.MIN_NUMERICAL_ITEMS, 
                                 max_num=settings.MAX_NUMERICAL_ITEMS, 
                                 current_num=None):
        
        if current_num == None:
            current_num = NumContainer()
        
        new_instance = cls([current_num.num])
        current_num.num += 1
        
        while (current_num.num < min_num) or (random.random() <= p_continue and current_num.num < max_num):
            if random.random() <= p_new_list:
                new_item = cls.generate_random_instance(p_continue, p_new_list, min_num, max_num, current_num)
            else:
                new_item = current_num.num
                current_num.num += 1
            
            new_instance.append(new_item)
        
        return new_instance
    
    def get_tokens(self):
        
        yield "["
        for item in self:
            if hasattr(item, "get_tokens"):
                for t in item.get_tokens():
                    yield t
            else:
                yield item
        yield "]"
    
    def calc_score(self, gold_standard):
        """
        See Klien (2005), 'The Unsupervised Learning Of Natural Language Structure', section 2.2.2.
        """
        brackets = self.get_brackets()
        gold_brackets = gold_standard.get_brackets() 
        klien_score = calc_f1_score(brackets, gold_brackets)
        
        descendents = set(self.get_descendents())
        gold_descendents = set(gold_standard.get_descendents())
        descendents_score = calc_f1_score(descendents, gold_descendents)
        
        return mean([klien_score, descendents_score])


    def get_descendents(self):
        
        for item in self:
            if hasattr(item, "get_descendents"):
                for t in item.get_descendents():
                    yield t
            else:
                yield item
    
    def get_brackets(self):
        
        result = set()
        
        for item in self:
            if hasattr(item, "get_brackets"):
                result |= item.get_brackets()
        
        descendents = list(self.get_descendents())
        
        if len(descendents) == 0:
            # Gold standard Nested lists generated with generate_random_instance never contain
            # empty sets. But the ANN can generate empty sets. In such a case we denote these
            # redundant brackets as (-1,-1) which never exists in the gold standard and hence 
            # reduces the f1 score of the network
            descendents.append(-1)
        
        result.add(  (min(descendents), max(descendents))  )
                
        return result
    
    def __hash__(self):
        return hash(tuple(self))
