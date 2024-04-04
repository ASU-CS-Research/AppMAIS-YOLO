import tensorflow as tf


a = tf.constant([1, 2, 3])
b = tf.constant([4, 5])
c = tf.ragged.constant([a,b])
print(c)