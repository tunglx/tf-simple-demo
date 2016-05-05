import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

product = tf.matmul(x, W)
y = product + b

y_ = tf.placeholder(tf.float32, [None, 1])
cost = tf.reduce_sum(tf.pow((y_ - y), 2))

train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

steps = 1000
for i in range(steps):
	xs = np.array([[i]])
	ys = np.array([[5*i]])
	feed = {x: xs, y_: ys}
	sess.run(train_step, feed_dict = feed)
	print("After %d iterations:" %i)
	print("W: %f" % sess.run(W))
	print("b: %f" % sess.run(b))
