import tensorflow as tf
import numpy as np

# Tạo dữ liệu giả đơn giản
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32) # Tập phát triển
target = np.array([[2.0, 4.0, 6.0, 8.0, 10.0]], dtype=np.float32) # Tập kiểm tra

# Xây dựng mô hình RNN đơn giản với Activation Function là tanh
input = tf.placeholder(tf.float32, shape=[1, 5])
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(activation=tf.tanh)
rnn_layer, _ = tf.nn.dynamic_rnn(rnn_cell, input, dtype=tf.float32, time_major=True)
output = tf.transpose(rnn_layer, [1, 0, 2])[-1]

# Định nghĩa Loss Function
loss = tf.reduce_mean(tf.square(tf.subtract(output, target)))

# Gradient Clipping
optimizer = tf.train.GradientDescentOptimizer(0.01)
gradients_and_vars = optimizer.compute_gradients(loss)
clipped_gradients_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients_and_vars]
train_op = optimizer.apply_gradients(clipped_gradients_and_vars)

# Khởi tạo phiên TensorFlow
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # Biên dịch và huấn luyện mô hình
    for i in range(100):
        _, loss_value = session.run([train_op, loss], feed_dict={input: data})
        print("Epoch {}, Loss: {}".format(i, loss_value))
