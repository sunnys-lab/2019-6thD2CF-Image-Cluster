"""
폴더와 임시 파일을 지정하는 Constants 들입니다.
"""

# dataset
#DATASET = "dummy"               # img 폴더 아래 폴더를 새로 만들고 해당 폴더이름으로 바꿔야 함
DATASET = "dev"               # img 폴더 아래 폴더를 새로 만들고 해당 폴더이름으로 바꿔야 함

# image format
IMG_EXT = "jpg"                 # 이미지 파일 형식

# output dirs
IMG_DIR = "./img/" + DATASET   # 이미지 파일 경로
DATA_DIR = "./data/" + DATASET # 중간 결과 경로

# files generated
IMG_PATHS = "img_paths.txt"     # 이미지 파일 리스트
LABELS_TRUE = "labels_true"     # 정답 레이블
LABELS_PRED = "labels_pred"     # 예측 레이블

# for clustering
NUM_IMGS_PER_MODEL = 70         # 클러스터당 평균 이미지수

# LEARNING_RATE
LEARNING_RATE = 0.01

# ETC
RANDOM_SEED = 2019
TRAIN_FRACTION = 0.7
GROUP_NAME_LENGTH = 10

# How long will we train the network (number of batches).
NUM_TRAIN_STEPS = 100 #@param {type: 'integer'}
# How many training examples we use in each step.
TRAIN_BATCH_SIZE = 10 #@param {type: 'integer'}
# How often to evaluate the model performance.
EVAL_EVERY = 10 #@param {type: 'integer'}

ITEM_PHOTOS_PATH = './item_photos'
