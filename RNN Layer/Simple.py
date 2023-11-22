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

# Khởi tạo phiên TensorFlow
with tf.Session() as session:
    # Biên dịch và huấn luyện mô hình
    loss_value = session.run(loss, feed_dict={input: data})

    # In ra giá trị mất mát
    print("Loss:", loss_value)
