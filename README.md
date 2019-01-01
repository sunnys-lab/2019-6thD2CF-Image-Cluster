># 2019 6th D2 CampusFest Image Clustering

- 6th D2 CampusFest Image Clustering 대회 참가작
- tensorflow-hub를 이용한 이미지 유사도 분석을 통하여 유사이미지들을 clustering
- 아래의 내용은 다음의 reference들을 참고하여 작성:  
 http://androidkt.com/retrain-image-classifier-model-using-tensorflow-hub/
 https://opensource.com/article/17/12/tensorflow-image-classification-part-1

&nbsp;
>## 03. Creating Training Set of Images
- 기존 이미지들을 tensorflow-hub 의 retrain.py에 사용하기 위해 class별 subfoler를 구성이 필요
- make_item_subfolers.py 를 실행하여 해당 작업을 자동화 

&nbsp; 
>## 04.Retrain Image Module

* Training 이미지 준비 후 최신 TensorFlow 버전(1.7*)과 TensorFlow Hub를 설치
* Training은 다음과 같이 실행

~~~
python3 retrain.py --image_dir data_dir \
--saved_model_dir saved_model_dir \
--bottleneck_dir bottleneck_dir \
--how_many_training_steps 2000 \
--output_labels output/output_labels.txt \
--output_graph output/retrain.pb
~~~

* 본 프로잭트에서는 다음과 같이 실행
~~~ 
python3 retrain.py --image_dir item_photos --how_many_training_steps 100 --output_graph ./output/output_graph.pb --output_labels ./output/output_labels.txt
~~~

* Training이 끝나면 다음과 같은 형태로 종료
~~~
INFO:tensorflow:2019-01-02 05:44:59.300157: Step 99: Validation accuracy = 100.0% (N=100)
2019-01-02 05:45:01.396277: W tensorflow/core/graph/graph_constructor.cc:1265] Importing a graph with a lower producer version 26 into an existing graph with producer version 27. Shape inference will have run different parts of the graph with different producer versions.
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Restoring parameters from /tmp/_retrain_checkpoint
INFO:tensorflow:Final test accuracy = 100.0% (N=32)
INFO:tensorflow:Save final result to : output_graph.pb
2019-01-02 05:45:06.444566: W tensorflow/core/graph/graph_constructor.cc:1265] Importing a graph with a lower producer version 26 into an existing graph with producer version 27. Shape inference will have run different parts of the graph with different producer versions.
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Restoring parameters from /tmp/_retrain_checkpoint
INFO:tensorflow:Froze 378 variables.
INFO:tensorflow:Converted 378 variables to const ops.
WARNING:tensorflow:From E:/PycharmProjects/2019-6thD2CF-Image-Cluster/retrain.py:908: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
~~~

As retrain.py proceeds, the training images are automatically separated into batches of training, test, and validation data sets.

In the output, we're hoping for high "Train accuracy" and "Validation accuracy" and low "Cross entropy." See How to retrain Inception's final layer for new categories for a detailed explanation of these terms. Expect training to take around 30 minutes on modern hardware.

Pay attention to the last line of output in your console:

INFO:tensorflow:Final test accuracy = 100.0% (N=32)
This says we've got a model that will, nine times out of 10, correctly guess which one of five possible flower types is shown in a given image. Your accuracy will likely differ because of randomness injected into the training process.

The script loads the pre-trained module and trains a new classifier on top of the fruits photos.You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it’s in.

The top layer receives as input a 2048-dimensional vector for each image. We train a softmax layer on top of this representation. If the softmax layer contains N labels, this corresponds to learning N + 2048*N model parameters for the biases and weights

&nbsp;
>## 5.TensorBoard  
You can visualize the graph and statistics, such as how the weights or accuracy varied during training.
Run this command during or after retraining.

~~~
tensorboard --logdir /tmp/retrain_logs
~~~

After TensorBoard is running, navigate your web browser to localhost:6006 to view the TensorBoard.

&nbsp;
>## 6.Testing Model
다음의 코드를 make_labels_pred.py 에 응용하여 prediction을 시행

```python
import tensorflow as tf, sys
 
image_path = sys.argv[1]
graph_path = './output_graph.pb'
labels_path = './output_labels.txt'
 
# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
 
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
    in tf.gfile.GFile(labels_path)]
 
# Unpersists graph from file
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
 
# Feed the image_data as input to the graph and get first prediction
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, 
    {'DecodeJpeg/contents:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
         human_string = label_lines[node_id]
         score = predictions[0][node_id]
         print('%s (score = %.5f)' % (human_string, score))
```

To test your own image, save it as test.jpg in your local directory and run (in the container) python classify.py test.jpg. The output will look something like this:
