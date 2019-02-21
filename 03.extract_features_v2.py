"""
MobileNet V2으로 이미지 특징 벡터를 추출하는 모듈입니다.
"""
import os
import time
import numpy as np
import tensorflow as tf
from mobilenet_v2 import MobileNet2, get_encoded_image
from amoebanet_v1 import AmoebaNet1, get_encoded_image
from pnasnet_v2 import PNasNet2, get_encoded_image
from nasnet_large import Nasnet_large, get_encoded_image
from inception_v3 import Inception_v3, get_encoded_image
from inception_resnet_v2 import Inception_ResNet_v2, get_encoded_image
from resnet_v2_152 import ResNet_v2_152, get_encoded_image
from config import *
import h5py


def extract_features():
    """
    IMG_DIR에 있는 모든 이미지에 대해 feature vector 추출
    추출된 특징 벡터는 DATA_DIR/FEATURES.npy 에 저장
    BATCH_SIZE로 배치 사이즈를 조절
    :return: 없음
    """
    # get list all images
    img_paths = os.listdir(IMG_DIR)
    img_paths.sort()
    img_paths = [os.path.join(IMG_DIR, filename) for filename in img_paths if filename.endswith(IMG_EXT)]
    with open(os.path.join(DATA_DIR, IMG_PATHS), 'w') as f:
        f.writelines([line + "\n" for line in img_paths])

    # prepare tf.dataset for batch inference
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda img_path:
                          (img_path, tf.py_func(get_encoded_image, [img_path], [tf.string])))
    batched_dataset = dataset.batch(BATCH_SIZE)
    iterator = batched_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    # build dnn model
    model = Inception_v3()
    #model.load_weights("final_weights.hdf5", by_name=True)

    # GPU 사용 옵션 추가 (2019.02.06)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # batch inference images
    num_processed_images = 0
    features = np.ndarray(shape=[0, model.output_size])
    while True:
        try:
            start_time = time.process_time()
            # get batch of encoded images
            batch = sess.run(next_batch)
            batched_img_paths = batch[0]
            batched_encoded_images = batch[1]
            cur_batch_size = len(batched_img_paths)
            # get batch of features
            batched_features = sess.run(model.features, feed_dict={
                model.filename: batched_img_paths,
                model.encoded_images: np.reshape(batched_encoded_images, [cur_batch_size])})
            features = np.concatenate((features, batched_features))
            # show progress
            num_processed_images += len(batched_encoded_images)
            elapsed_time = time.process_time() - start_time
            print("Processed images: %d,\tElapsed time: %.2f,\tElapsed time per img: %.2f" % (num_processed_images, elapsed_time, elapsed_time / BATCH_SIZE))
        except tf.errors.OutOfRangeError:
            break

    # save npy and tsv files
    if os.path.exists(DATA_DIR) is False:
        os.makedirs(DATA_DIR)
    np.save(os.path.join(DATA_DIR, FEATURES + ".npy"), features)
    np.savetxt(os.path.join(DATA_DIR, FEATURES + ".tsv"), features, delimiter="\t")



if __name__ == '__main__':
    extract_features()
