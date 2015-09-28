#===============================================================================
# Imports
#===============================================================================

import copy
import logging

import numpy as np
import scipy.stats

from utils import ShallowCopyProxy
from NestedList import NestedList, NestedListManager


#===============================================================================
# Classes
#===============================================================================

class FeedForwardProcess(object):
    
    def __init__(self, network, input_struct=None):
        
        # Deep-copying keras models is very expensive. We avoid this by specifying that network 
        # attribute should always be shallow copied.
        self._network = ShallowCopyProxy(network)
        
        if input_struct == None:
            input_struct = NestedList.generate_random_instance()
        self.input_struct = input_struct
        
        self.tokens = list(input_struct.get_tokens())
        
        self.output_struct = NestedListManager()
        self.execution_history = []
    
    @property
    def network(self):
        return self._network.referent
    
    def do_next_step(self):
        
        token = self.tokens[len(self.execution_history)]
        
        output_layer, action = self.network.feed_forward(token)
        
        step = FeedForwardStep(token, copy.deepcopy(self.output_struct), output_layer, action)
        self.execution_history.append(step)
        self.output_struct.do_action(action, token)
        
    def run_to_end(self):
        
        while len(self.execution_history) < len(self.tokens):
            self.do_next_step()
    
    def calc_total_score(self):
        
        set(self.input_struct.get_descendents())
        
        self.output_struct.root.get_descendents()
        
    
    def calc_score(self):
        # Wrap the input_struct in an additional NestedList because the NestedListManager for output_strcut 
        # automatically initializes input_struct.root to be a NestedList and does everything inside that root.
        return self.output_struct.root.calc_score(NestedList(self.input_struct))
    
    def fork(self, step_num):
        new_instance = copy.deepcopy(self)
        new_instance.output_struct = new_instance.execution_history[step_num].struct
        new_instance.execution_history = new_instance.execution_history[:step_num]
        return new_instance
    
    def get_random_fork(self):
        
        logging.log(logging.NOTSET, "Choosing at which step to fork")
        entropies = [step.entropy for step in self.execution_history]
        total_entropy = sum(entropies)
        probabilities = [e/total_entropy for e in entropies]
        step_index = np.random.choice(len(self.execution_history), p=probabilities)
        step = self.execution_history[step_index]
        
        logging.log(logging.NOTSET, "Choosing how to modify this step")
        probabilities = step.network_output
        probabilities[step.action] = 0
        probabilities = probabilities * (1/sum(probabilities))
        
        num_possible_actions = len(step.network_output)
        
        new_action = np.random.choice(num_possible_actions, p=probabilities)
        
        new_network_output = self.network.encode_action(new_action)
        
        logging.log(logging.NOTSET, "Forking")
        fork = self.fork(step_index)
        
        logging.log(logging.NOTSET, "Applying modification")
        new_step = FeedForwardStep(step.token, copy.deepcopy(step.struct), new_network_output, new_action)
        fork.execution_history.append(new_step)
        fork.output_struct = copy.deepcopy(step.struct)
        fork.output_struct.do_action(new_action, step.token)
        
        return fork, step_index


class FeedForwardStep(object):
    def __init__(self, token, struct, network_output, action):
        self.token = token
        self.struct = struct
        self.network_output = network_output 
        self.action = action
        self.entropy = scipy.stats.entropy(network_output)  
