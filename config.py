"""
폴더와 임시 파일을 지정하는 Constants 들입니다.
"""

# dataset
DATASET = "train"                        # img 폴더 아래 폴더를 새로 만들고 해당 폴더이름으로 바꿔야 함
                                         # train: 실제 테스트용으로 전체 이미지 넣을것
                                         # dev: 개발용으로 일부 이미지만 넣을것

# image format
IMG_EXT = "jpg"                         # 이미지 파일 형식

# output dirs
RAW_IMG_DIR = "./raw_img/" + DATASET   # 원본 이미지 파일 경로
IMG_DIR = "./img/" + DATASET           # 전처리 완료한 이미지 파일 경로
DATA_DIR = "./data/" + DATASET         # 중간 결과 경로
CLUSTER_DIR = "./result/" + DATASET    # 클러스터링 결과 경로

# files generated
IMG_PATHS = "img_paths.txt"             # 이미지 파일 경로
FEATURES = "features"                   # 이미지 특징 벡터
SIGNATURES = "signature"                # 이미지 시그니쳐
LABELS_TRUE = "labels_true"             # 정답 레이블
LABELS_PRED = "labels_pred"             # 예측 레이블

# for mobilenet feature extraction
BATCH_SIZE = 128                        # batch size

# for clustering
NUM_IMGS_PER_MODEL = 70                 # 클러스터당 평균 이미지수


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

FINE_TRAIN_IMG_PATH = './training_images'


