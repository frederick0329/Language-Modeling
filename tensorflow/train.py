import sys
sys.path.append("..")
import numpy as np
import os
from ptb import *
from rnnlm import *
from logger import *

gpu_number = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
tf.set_random_seed(24)
    
class Trainer():
    def __init__(self, train_batch_size=20, train_seq_len=35, test_batch_size=10, test_seq_len=35, exp_name='exp'):
        self.dataset = PTB(train_batch_size, test_batch_size, train_seq_len, test_seq_len)
        self.model = RNNLM(self.dataset.vocabulary_size, 200, 200, 2)
        self.build_graph()
        self.logger = Logger(exp_dir=exp_name)
        self.tensorboard_path = '/tmp/rnnlm/'+ exp_name

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # session config
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True

            # input placeholders
            self.inputs_placeholder = tf.placeholder(
            	tf.int64,
            	shape = [None, None],
            	name = 'inputs'
            )
            self.targets_placeholder = tf.placeholder(
                tf.int64,
                shape = [None, None],
                name = 'targets'
            )

            # learning rate placeholder
            self.lr_placeholder = tf.placeholder(
                tf.float32,
                shape = (),
                name = 'lr'
            )

            self.hidden_state_placeholder = tf.placeholder(
               tf.float32,
               shape = [2, 2, None, 200],
               name = 'hidden'
            )
            
            # network          
            self.logits, self.hidden_state = self.model.build_network(self.inputs_placeholder, self.hidden_state_placeholder)
            
            # loss
            weights = tf.ones_like(self.targets_placeholder, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets_placeholder, weights, average_across_timesteps=True)

            # optimizer
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(self.lr_placeholder)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.25)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            
            # train summary
            # loss 
            self.train_loss = tf.placeholder(tf.float32, shape=())
            self.train_loss_summary = tf.summary.scalar('train_loss', self.train_loss)
            # acc 
            self.train_perplexity_value = tf.placeholder(tf.float32, shape=())
            self.train_perplexity_summary = tf.summary.scalar('train_perplexity', self.train_perplexity_value)
            
            # test summary
            # loss 
            self.test_loss = tf.placeholder(tf.float32, shape=())
            self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
            # acc 
            self.test_perplexity_value = tf.placeholder(tf.float32, shape=())
            self.test_perplexity_summary = tf.summary.scalar('test_perplexity', self.test_perplexity_value)

    def train(self, epochs=1):
        with self.graph.as_default():
            writer = tf.summary.FileWriter(self.tensorboard_path, graph=tf.get_default_graph())
            with tf.Session(config=self.config) as sess:
                saver = tf.train.Saver(max_to_keep=100)  
                best_test_loss = None
                lr = 20.0
                all_initializer_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(all_initializer_op)
                for i in range(epochs):
                    hidden_state = self.model.init_hidden(self.dataset.train_batch_size) 
                    total_loss = 0.0
                    for j in range(self.dataset.train_batch_count):
                        batch_inputs, batch_targets = self.dataset.next_train_batch(j)
                        loss, logits, hidden_state, _ = sess.run([self.loss, self.logits, self.hidden_state, self.optim], 
                                                                 feed_dict = {self.inputs_placeholder : batch_inputs, 
                                                                              self.targets_placeholder : batch_targets,
                                                                              self.hidden_state_placeholder : self.tuple_to_np(hidden_state),
                                                                              self.lr_placeholder : lr})
                        total_loss += loss
                    avg_loss = total_loss / self.dataset.train_batch_count 
                    avg_perplexity =  np.exp(avg_loss)
                    # logging training results
                    summary = sess.run(self.train_loss_summary, feed_dict={self.train_loss: avg_loss})
                    writer.add_summary(summary, i)
                    summary = sess.run(self.train_perplexity_summary, feed_dict={self.train_perplexity_value: avg_perplexity})
                    writer.add_summary(summary, i) 
                    self.logger.log('Training epoch {0}, learning rate {1}'.format(i, lr))
                    self.logger.log('    train loss {0}, train perplexity {1}'.format(avg_loss, avg_perplexity))
                    total_loss = 0.0
                    # evaluate on test set
                    hidden_state = self.model.init_hidden(self.dataset.test_batch_size)  
                    for j in range(self.dataset.test_batch_count):
                        batch_inputs, batch_targets = self.dataset.next_test_batch(j)
                        loss, logits, hidden_state = sess.run([self.loss, self.logits, self.hidden_state], 
                                                                 feed_dict = {self.inputs_placeholder : batch_inputs, 
                                                                              self.targets_placeholder : batch_targets,
                                                                              self.hidden_state_placeholder : self.tuple_to_np(hidden_state)})
                        total_loss += loss
                    avg_loss = total_loss / self.dataset.test_batch_count
                    avg_perplexity = np.exp(avg_loss)
                    
                    if best_test_loss is not None and best_test_loss < avg_loss:
                        # Anneal the learning rate if no improvement has been seen in the validation dataset.
                        lr /= 4.0
                    elif best_test_loss is None or best_test_loss > avg_loss:
                        best_test_loss = avg_loss

                    # logging validation results
                    summary = sess.run(self.test_loss_summary, feed_dict={self.test_loss: avg_loss})
                    writer.add_summary(summary, i)
                    summary = sess.run(self.test_perplexity_summary, feed_dict={self.test_perplexity_value: avg_perplexity})
                    writer.add_summary(summary, i)
                    self.logger.log('    test loss {0}, test perplexity {1}'.format(avg_loss, avg_perplexity))
                    # save model
                    save_model_file = os.path.join(self.logger.exp_dir, 'RNNLM-model')
                    if i % 20 == 0:
                        saver.save(sess, save_model_file, global_step=self.global_step)
    
    def tuple_to_np(self, t):
        return np.stack(t)
if __name__ == "__main__":
    trainer = Trainer(exp_name='exp1')
    trainer.train(epochs=100)
