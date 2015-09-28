#===============================================================================
# Intro to computational neuroscience
# Term paper
# Jonathan Shafer
# September 2015 
#===============================================================================

#===============================================================================
# Imports
#===============================================================================

import datetime
import logging
import os

import numpy as np

from keras.optimizers import SGD

import settings
import NestedList
from FeedForwardProcess import FeedForwardProcess
from Network import Network


#===============================================================================
# Constants
#===============================================================================

PUNCTUIATION_TOKENS = ["[","]"]
NUM_PUNCTUATION_TOKENS = len(PUNCTUIATION_TOKENS)
INPUT_VECTOR_SIZE = settings.MAX_NUMERICAL_ITEMS + NUM_PUNCTUATION_TOKENS 
HIDDEN_LAYER_SIZE = INPUT_VECTOR_SIZE
NUM_ACTIONS = 3


#===============================================================================
# Functions
#===============================================================================

def evaluate_network(network, corpus):
    scores = []
    for instance in corpus:
        process = FeedForwardProcess(network, instance)
        process.run_to_end()
        scores.append(process.calc_score())
    return np.mean(scores)


#===============================================================================
# Main
#===============================================================================

# Set up logging
if not os.path.exists(settings.LOGS_DIRECTORY):
    os.makedirs(settings.LOGS_DIRECTORY)
logging.basicConfig(level=logging.DEBUG, filename= os.path.join(settings.LOGS_DIRECTORY, '%s.log' % datetime.datetime.now().isoformat()))
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logging.getLogger().addHandler(sh)


logging.info("Creating network")
network = Network(settings.MAX_NUMERICAL_ITEMS, PUNCTUIATION_TOKENS, NUM_ACTIONS)
network.model.compile(loss="mean_squared_error", optimizer=SGD(lr=5))


logging.info("Generating corpus")
corpus = NestedList.generate_corpus(settings.CORPUS_SIZE)


logging.info("Evaluating untrained network")
mean_f1_score = evaluate_network(network, corpus)
logging.info("Mean score: %f", mean_f1_score)


logging.info("Training the network")
for i in range(settings.NUM_EPOCHS):
    
    logging.info("Epoch %d", i)
    
    process = FeedForwardProcess(network)
    logging.debug("Initial input: %s", [process.input_struct])
    process.run_to_end()
    logging.debug("Initial output: %s", process.output_struct.root)
    score = process.calc_score()
    logging.debug("Initial score: %f", score)
    
    if score != 1:
        
        fork_score = 0
        fork_count = 0
        
        while fork_score <= score and fork_count < settings.MAX_FORK_ATTEMPTS:
            logging.debug("Fork attempt %d", fork_count)
            fork, fork_step_index = process.get_random_fork()
            logging.debug("Running feed forward") 
            fork.run_to_end()
            logging.debug("Fork output: %s", fork.output_struct.root)
            fork_score = fork.calc_score()
            fork_count += 1
            logging.debug("Fork candidate score: %f", fork_score)
        
        if fork_score > score:
            logging.debug("Better variation found. Fitting network.")
            fork_step = fork.execution_history[fork_step_index]
            network.fit(fork_step.token, fork_step.action)
        
        else:
            logging.debug("No better variation was found.")
            

logging.info("Training complete. Evaluating trained network")
mean_f1_score = evaluate_network(network, corpus)
logging.info("Mean score: %f", mean_f1_score)
