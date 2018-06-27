
# Create model


def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, patch_size, patch_size, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



# Store layers weight & bias
weights = {
    # 4x4 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 4x4 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 16*16*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(16)*(16)*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

with tf.name_scope('model'):
    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    print(logits.name)
    #logits = Lenet(X, lenet_weights, lenet_biases, keep_prob)
    prediction = tf.nn.softmax(logits)
    print(prediction.name)
with tf.name_scope('loss'):
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

with tf.name_scope('Accuracy'):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tf.summary.scalar("loss", loss_op)

tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
    for step in range(1, training_epochs+1):
        #avg_cost=0
        total_batch = int(train_all.shape[0]/batch_size)
        #for i in range(total_batch):
        batch_x, batch_y = model.next_batch(batch_size)
        # Run optimization op (backprop)
        _, summary = sess.run([train_op, merged_summary_op], feed_dict={
                              X: batch_x, Y: batch_y, keep_prob: dropout})
        #write logs
        summary_writer.add_summary(summary, step)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            # Compute average loss
            #avg_cost += loss / total_batch
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    saver.save(sess, './trained-models/cnn/my_cnn_model_final')
    # Calculate accuracy for test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_all,
                                        Y: labels_all_test,
                                        keep_prob: 1.0}))
