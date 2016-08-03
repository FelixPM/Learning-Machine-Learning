import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=10000)

# x = tf.constant(35, name='x')
# x = tf.constant([35, 40, 45], name='x')
x = tf.constant(data, name='x')
y = tf.Variable(5*x**2-3*x+15, name='y')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(x))
    print(session.run(y))

