#===============================================================================
# Imports
#===============================================================================

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation


#===============================================================================
# Classes
#===============================================================================

class Network(object):
    
    def __init__(self, max_numerical_items, punctuation_tokens, num_actions, hiddern_layer_size=None ):
        
        self.max_numerical_items = max_numerical_items
        self.punctuation_tokens = punctuation_tokens
        self.input_layer_size = max_numerical_items + len(punctuation_tokens)
        
        self.hiddern_layer_size = hiddern_layer_size
        if hiddern_layer_size == None:
            self.hiddern_layer_size = self.input_layer_size
        
        self.num_actions = num_actions
        
        self.model = Sequential()
        self.model.add(Dense(self.input_layer_size, self.hiddern_layer_size, init="uniform"))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(self.hiddern_layer_size, self.num_actions, init="uniform"))
        self.model.add(Activation("softmax"))
    
    def feed_forward(self, token):
        input_layer = self.encode_token(token)
        
        output_layer = self.model.predict(np.array([input_layer]))[0]
        
        action = self.decode_action(output_layer)
        return output_layer, action
    
    def fit(self, token, action):
        input_layer = self.encode_token(token)
        required_output_layer = self.encode_action(action)
        self.model.fit(np.array([input_layer]), np.array([required_output_layer]), verbose=False)
    
    def encode_token(self, token):
        value = token 
        if token in self.punctuation_tokens:
            value = self.max_numerical_items + self.punctuation_tokens.index(token)
        
        assert isinstance(value, int), "Cannot encode token '%s'" % token
        
        return self.one_hot_encoding(value, self.input_layer_size)
    
    def decode_action(self, output_layer):
        return [int(value == max(output_layer)) for value in output_layer].index(1)
    
    def encode_action(self, action):
        return self.one_hot_encoding(action, self.num_actions)
        
    def one_hot_encoding(self, value, max_value):
        return [int(i == value) for i in range(max_value)]
