import numpy as np
from layers import *
import tensorflow as tf

class RNNLM():
    def __init__(self, vocabulary_size, embedding_size, hidden_units, num_layers):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers 

    def build_network(self, input_words, hidden_state):
        with tf.variable_scope('RNNLM'):
            batch_size = tf.shape(input_words)[0]
            time_steps = tf.shape(input_words)[1]
            embed = embedding('embedding', input_words, self.vocabulary_size, self.embedding_size)
            layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_units), input_keep_prob=0.8, output_keep_prob=0.8)]
            for i in range(1, self.num_layers):
                layers.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_units), output_keep_prob=0.8))
            rnn_layers = tf.nn.rnn_cell.MultiRNNCell(layers)
            self.rnn_layers = rnn_layers
            l = tf.unstack(hidden_state, axis=0)
            rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1]) for idx in range(self.num_layers)])
            outputs, hidden_state = tf.nn.dynamic_rnn(rnn_layers, embed, initial_state=rnn_tuple_state)
           
            outputs = tf.reshape(outputs, [batch_size * time_steps, rnn_layers.output_size])
            outputs = fully_connected('fc', outputs, rnn_layers.output_size, self.vocabulary_size)            
            outputs = tf.reshape(outputs, [batch_size, time_steps, self.vocabulary_size])

            return outputs, hidden_state

    def init_hidden(self, batch_size):
         return tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros([batch_size, self.hidden_units]), 
                                                     np.zeros([batch_size, self.hidden_units])) for idx in range(self.num_layers)])
