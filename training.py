# -*- coding: utf-8 -*-

import os
import random
import json
import collections
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from config import *
from IPython.display import clear_output, Image, display, HTML


def make_train_and_test_sets():
  """Split the data into train and test sets and get the label classes."""
  train_items, test_items = [], []
  group_names = []
  shuffler = random.Random(RANDOM_SEED)

  file_names = os.listdir(IMG_DIR)
  file_names.sort()

  #Extract class name from file names - 파일명의 일부분이 class name임으로 어떤 class가 있는지 추출
  for i in range(len(file_names)):
    group_names.append(file_names[i][0:GROUP_NAME_LENGTH])

  # class 리스트 자료형에서 중복되는 그룹명 삭제 및 정리
  class_names = list(set(group_names))
  class_names.sort()
  classes = collections.OrderedDict(enumerate(class_names))

  #print(file_names)
  #print(class_names)

  # Making dicionary - 파일명을 읽어서 단순 index값을 가지는 dictionary 자료형 생성
  class_to_index = dict([(x, i) for i, x in enumerate(class_names)])

  # CONVERT dictionary to json using json.dump & Write JSON
  with open('dictionary.json', 'w', encoding="utf-8") as make_file:
      json.dump(class_to_index, make_file, ensure_ascii=False, indent="\t")

  items = []
  for j in range(len(file_names)):
      item = (IMG_DIR + '/' + file_names[j],class_to_index[file_names[j][0:GROUP_NAME_LENGTH]])
      items.append(item)

  num_train = int(len(file_names) * TRAIN_FRACTION)
  train_items.extend(items[:num_train])
  test_items.extend(items[num_train:])

  #print(train_items)
  #print(test_items)
  shuffler.shuffle(train_items)
  shuffler.shuffle(test_items)
  return train_items, test_items, classes


def decode_and_resize_image(encoded):
  decoded = tf.image.decode_jpeg(encoded, channels=3)
  decoded = tf.image.convert_image_dtype(decoded, tf.float32)
  return tf.image.resize_images(decoded, image_size)


def create_model(features):
  """Build a model for classification from extracted features."""
  # Currently, the model is just a single linear layer. You can try to add
  # another layer, but be careful... two linear layers (when activation=None)
  # are equivalent to a single linear layer. You can create a nonlinear layer
  # like this:
  # layer = tf.layers.dense(inputs=..., units=..., activation=tf.nn.relu)
  layer = tf.layers.dense(inputs=features, units=NUM_CLASSES, activation=None)
  return layer



def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for node in graph_def.node:
        stripped_node = strip_def.node.add()
        stripped_node.MergeFrom(node)
        if stripped_node.op == 'Const':
            tensor = stripped_node.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def



def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


#@title Show some labeled images
def get_label(example):
  """Get the label (number) for given example."""
  return example[1]


def get_class(example):
  """Get the class (string) of given example."""
  return CLASSES[get_label(example)]


def get_encoded_image(example):
  """Get the image data (encoded jpg) of given example."""
  image_path = example[0]
  #return tf.gfile.FastGFile(image_path, 'rb').read()  #python2
  return tf.gfile.GFile(image_path, 'rb').read()   #python3


def get_image(example):
  """Get image as np.array of pixels for given example."""
  #return plt.imread(StringIO(get_encoded_image(example)), format='jpg')  #python2
  return plt.imread(BytesIO(get_encoded_image(example)), format='jpg')   #python3


def display_images(images_and_classes, cols=5):
  """Display given images and their labels in a grid."""
  rows = int(math.ceil(len(images_and_classes) / cols))
  fig = plt.figure()
  fig.set_size_inches(cols * 3, rows * 3)
  for i, (image, flower_class) in enumerate(images_and_classes):
    plt.subplot(rows, cols, i + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title(flower_class)
  plt.show()


def get_batch(batch_size=None, test=False):
  """Get a random batch of examples."""
  examples = TEST_ITEMS if test else TRAIN_ITEMS
  batch_examples = random.sample(examples, batch_size) if batch_size else examples
  return batch_examples


def get_images_and_labels(batch_examples):
  images = [get_encoded_image(e) for e in batch_examples]
  one_hot_labels = [get_label_one_hot(e) for e in batch_examples]
  return images, one_hot_labels


def get_label_one_hot(example):
  """Get the one hot encoding vector for the example."""
  one_hot_vector = np.zeros(NUM_CLASSES)
  np.put(one_hot_vector, get_label(example), 1)
  return one_hot_vector




if __name__ == '__main__':
    ### 01. 기존 이미지를 이용한 dictionary 생성 및 데이터에 대한 기본 정보 출력
    TRAIN_ITEMS, TEST_ITEMS, CLASSES = make_train_and_test_sets()
    NUM_CLASSES = len(CLASSES)
    print('\nThe dataset has %d label classes: %s' % (NUM_CLASSES, CLASSES.values()))
    print('There are %d training images' % len(TRAIN_ITEMS))
    print('there are %d test images' % len(TEST_ITEMS))

    ### 02. Training
    tf.reset_default_graph()

    # Load a pre-trained TF-Hub module for extracting features from images. We've
    # chosen this particular module for speed, but many other choices are available.
    image_module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2')

    # Preprocessing images into tensors with size expected by the image module.
    encoded_images = tf.placeholder(tf.string, shape=[None])
    image_size = hub.get_expected_image_size(image_module)

    batch_images = tf.map_fn(decode_and_resize_image, encoded_images, dtype=tf.float32)

    # The image module can be applied as a function to extract feature vectors for a
    # batch of images.
    features = image_module(batch_images)

    logits = create_model(features)
    labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Mathematically, a good way to measure how much the predicted probabilities
    # diverge from the truth is the "cross-entropy" between the two probability
    # distributions. For numerical stability, this is best done directly from the
    # logits, not the probabilities extracted from them.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # Let's add an optimizer so we can train the network.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss=cross_entropy_mean)

    # The "softmax" function transforms the logits vector into a vector of
    # probabilities: non-negative numbers that sum up to one, and the i-th number
    # says how likely the input comes from class i.
    probabilities = tf.nn.softmax(logits)

    # We choose the highest one as the predicted class.
    prediction = tf.argmax(probabilities, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))

    # The accuracy will allow us to eval on our test set.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Show the created TF graph.
    show_graph(tf.get_default_graph())

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(NUM_TRAIN_STEPS):
            # Get a random batch of training examples.
            train_batch = get_batch(batch_size=TRAIN_BATCH_SIZE)
            batch_images, batch_labels = get_images_and_labels(train_batch)
            # Run the train_op to train the model.
            train_loss, _, train_accuracy = sess.run(
                [cross_entropy_mean, train_op, accuracy],
                feed_dict={encoded_images: batch_images, labels: batch_labels})
            is_final_step = (i == (NUM_TRAIN_STEPS - 1))
            if i % EVAL_EVERY == 0 or is_final_step:
                # Get a batch of test examples.
                test_batch = get_batch(batch_size=None, test=True)
                batch_images, batch_labels = get_images_and_labels(test_batch)
                # Evaluate how well our model performs on the test set.
                test_loss, test_accuracy, test_prediction, correct_predicate = sess.run(
                    [cross_entropy_mean, accuracy, prediction, correct_prediction],
                    feed_dict={encoded_images: batch_images, labels: batch_labels})
                print('Test accuracy at step %s: %.2f%%' % (i, (test_accuracy * 100)))
        #모델 저장
        saver.save(sess, './model/my_test_model')




