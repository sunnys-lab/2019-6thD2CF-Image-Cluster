# -*- coding: utf-8 -*-

################################################
## Item photo를 group별로 subdirectory화 작업 ##
################################################
import os
import glob
import shutil

from config import *

group_names = []

def make_train_and_test_sets():
  """Split the data into train and test sets and get the label classes."""
  file_names = os.listdir(IMG_DIR)
  file_names.sort()

  #Extract class name from file names - 파일명의 일부분이 class name임으로 어떤 class가 있는지 추출
  for i in range(len(file_names)):
    group_names.append(file_names[i][0:GROUP_NAME_LENGTH])

  # class 리스트 자료형에서 중복되는 그룹명 삭제 및 정리
  class_names = list(set(group_names))
  class_names.sort()

  #print(file_names)
  #print(class_names)

  for names in class_names:
      dest_path = os.path.join(FINE_TRAIN_IMG_PATH, names)
      if not (os.path.isdir(dest_path)):
          os.makedirs(dest_path)
      target = IMG_DIR + '/' + names + '*.*'
      files = glob.glob(target)
      for file in files:
          shutil.copy(file,dest_path)

if __name__ == '__main__':
    make_train_and_test_sets()


