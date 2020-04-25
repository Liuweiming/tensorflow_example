import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as tfk
tfkl = tf.keras.layers

def example_1():

    # Two simple inputs
    intput_ = tf.placeholder(tf.float32, shape=(1, 100), name="input")
    target_ = tf.placeholder(tf.float32, shape=(1, 3), name="target")

    # Output
    output_ = tfkl.Activation("relu")(tfkl.Dense(64)(input_))
    output_ = tfkl.Activation("relu")(tfkl.Dense(3)(output_))
    output_ = tf.identity(output_, name="output")
    
    loss = 0.5 * tf.reduce_mean(tf.squared_difference(input_, target_))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, name='train')

    init = tf.variables_initializer(
        tf.global_variables(), name='init')

    # Write the model definition
    with open('load_model.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())


if __name__ == "__main__":
    example_1()