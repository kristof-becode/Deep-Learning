# Import `tensorflow`
#import tensorflow.compat.v1 as tf
import tensorflow as tf

#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Intialize the Session
sess = tf.compat.v1.Session()


# Print the result
print(result)

# Close the session
sess.close()


# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Initialize Session and run `result`
with tf.compat.v1.Session() as sess:
  output = sess.run(result)
  print(output)


config=tf.compat.v1.ConfigProto(log_device_placement=True)

config=tf.compat.v1.ConfigProto(allow_soft_placement=True)

