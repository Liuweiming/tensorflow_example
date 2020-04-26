import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as tfk
tfkl = tf.keras.layers


def example_1():

    with tf.Graph().as_default():
        # Two simple inputs
        a = tf.placeholder(tf.float32, shape=(1, 100), name="input_a")
        b = tf.placeholder(tf.float32, shape=(1, 100), name="input_b")

        # Output
        c = tf.add(a, b, name='result')

        # To add an init operation to the model
        i = tf.initializers.global_variables()

        # Write the model definition
        with open('loading.pb', 'wb') as f:
            f.write(tf.get_default_graph().as_graph_def().SerializeToString())


def example_2():

    with tf.Graph().as_default():
        # inputs
        input_ = tf.placeholder(tf.float32, shape=(1, 100), name="input")
        target_ = tf.placeholder(tf.float32, shape=(1, 3), name="target")

        # Output
        output_ = tfkl.Activation("relu")(tfkl.Dense(64)(input_))
        output_ = tfkl.Dense(3)(output_)
        output_ = tf.identity(output_, name="output")

        loss = 0.5 * tf.reduce_mean(tf.squared_difference(output_, target_))
        loss = tf.identity(loss, name="loss")

        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(loss, name="train")

        init = tf.variables_initializer(
            tf.global_variables(), name="init")

        # Write the model definition
        with open('training.pb', 'wb') as f:
            f.write(tf.get_default_graph().as_graph_def().SerializeToString())


if __name__ == "__main__":
    example_1()
    example_2()
