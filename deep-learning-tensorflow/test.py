hidden_layer1_size = 1024
hidden_layer2_size = 1024

hidden1_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer1_size]))
hidden1_biases = tf.Variable(tf.zeros([hidden_layer1_size]))
hidden1_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden1_weights) + hidden1_biases)

output_weights = tf.Variable(tf.truncated_normal([1024, num_labels]))
output_biases = tf.Variable(tf.zeros([num_labels]))
logits = tf.matmul(hidden2_layer, output_weights) + output_biases

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

train_prediction = tf.nn.softmax(logits)

# Setup validation prediction step.
valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, w1) + b1)
valid_logits = tf.matmul(valid_hidden, output_weights) + output_biases
valid_prediction = tf.nn.softmax(valid_logits)

# And setup the test prediction step.
test_hidden1 = tf.nn.relu(tf.matmul(tf_test_dataset, hidden1_weights) + hidden1_biases)
test_logits = tf.matmul(test_hidden2, output_weights) + output_biases
test_prediction = tf.nn.softmax(test_logits)

num_steps = 6001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in xrange(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step", step, ":", l)
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))






batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Initialize weight/biases.
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
