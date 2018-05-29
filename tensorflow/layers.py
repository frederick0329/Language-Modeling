import tensorflow as tf

def embedding(scope, inputs, vocabulary_size, embedding_size):
    with tf.variable_scope(scope):
        embedding = tf.Variable(tf.random_uniform((vocabulary_size, embedding_size), -0.1, 0.1),
                                dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, inputs)
        return embedded_input

def fully_connected(scope, input_layer, input_dim, output_dim):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        fc_weight = tf.get_variable(
            'fc_weight',
            shape = [input_dim, output_dim],
            dtype = tf.float32,
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
        )

        fc_bias = tf.get_variable(
            'fc_bias',
            shape = [output_dim],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        output_layer = tf.matmul(input_layer, fc_weight) + fc_bias

        return output_layer


