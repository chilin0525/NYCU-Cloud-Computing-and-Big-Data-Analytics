from typing import List
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def most_common(lst):
    lst = Counter(lst)
    return lst.most_common(1)[0][0]


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def read_mp4(mp4_filename: str)-> List[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4_filename))
    frames = list()
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # if frame length < SEQ_LENGTH, padding by repeat image
    more_40_flag = frame_length>40
    skip_num = int(frame_length/40)-1

    if skip_num>=0:
        skip_mod = frame_length%40
        if skip_mod<=0:
            skip_cnt = skip_num
        else:
            skip_cnt = skip_num + 1
            skip_mod -= 1
    else: 
        more_40_flag = False
        
    while cap.isOpened():
        ret, frame = cap.read()

        # End of video or error occured
        if not ret:
            break

        if more_40_flag:
            if skip_cnt>0:
                skip_cnt -= 1
                continue
        
        if more_40_flag:
            if skip_mod<=0:
                skip_cnt = skip_num
            else:
                skip_cnt = skip_num + 1
                skip_mod -= 1            

        # print(ret, frame/255.0, frame.shape, type(frame))
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, (224, 224))
        frame = frame[:, :, [2, 1, 0]]

        frames.append(frame/255.0)

    cap.release()
    cv2.destroyAllWindows()

    # if number of frame less then require seq length
    # repeating last image
    last_frame = frames[-1]
    while len(frames)<40:
        frames.append(last_frame)

    frames = np.array(frames)
    return frames


def get_mp4_frames(mp4_filename: str)-> int:
    """
    Get total number frame of mp4
    """
    cap = cv2.VideoCapture(str(mp4_filename))
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return frame_length


def data_generator(filename_list: List[str], batch_size=1):
    """Generator for read mp4 filename from training or testing folder,
    would yield each frame data in video and ground true label(class)
    """
    idx = 0
    while idx<len(filename_list):
        pixel_data_list = list()
        ground_true_list = list()
        for j in range(batch_size):
            if idx>=len(filename_list):
                break
            mp4_filename, ground_true = filename_list[idx]
            mp4_filename = mp4_filename.decode('ascii')
            pixel_data = read_mp4(mp4_filename)
            pixel_data_list.append(pixel_data)
            ground_true_list.append(int(ground_true))
            idx += 1
        yield (pixel_data_list, ground_true_list)


def testdata_generator(filename_list: List[str], batch_size=1):
    """Generator for read mp4 filename from training or testing folder,
    would yield each frame data in video and ground true label(class)
    """
    idx = 0
    while idx<len(filename_list):
        pixel_data_list = list()
        for j in range(batch_size):
            if idx>=len(filename_list):
                break
            mp4_filename = filename_list[idx]
            mp4_filename = mp4_filename.decode('ascii')
            pixel_data = read_mp4(mp4_filename)
            pixel_data_list.append(pixel_data)
            idx += 1
        yield pixel_data_list



train_dir_path = "/home/chilin/NYCU-Cloud-Computing-and-Big-Data-Analytics/HW1/train"
test_dir_path = "/home/chilin/NYCU-Cloud-Computing-and-Big-Data-Analytics/HW1/test"

# list all dir under train folder
train_dirs = os.listdir(train_dir_path)
test_dirs = os.listdir(test_dir_path)
# list to store all training mp4 file information
train_files = list()
test_files = list()
# read all training mp4 filename, also record class name
# all information would be a list of tuple(mp4 filepath, class)

test_files = [os.path.join(test_dir_path, i) for i in os.listdir(test_dir_path)]
# test_files = [test_files[0], test_files[1]]

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                            input_shape = (40, 224, 224, 3)))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((4, 4)))) 
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), padding='same',activation = 'relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((4, 4))))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))))
#model.add(TimeDistributed(Dropout(0.25)))
                                    
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())) 
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(39, activation = 'softmax'))

print(model.summary())

# split files in each class folder into training and testing part
train_files = list()
valid_files = list()
for class_dir in train_dirs:
    class_dir_files = os.listdir(f"{train_dir_path}/{class_dir}")
    train, valid = train_test_split(
        class_dir_files,
        test_size=0.2,
        random_state=123
    )   
    train_files.extend([(f"{train_dir_path}/{class_dir}/{i}", class_dir) for i in train])
    valid_files.extend([(f"{train_dir_path}/{class_dir}/{i}", class_dir) for i in valid])

print(len(train_files))
print(len(valid_files))
print(train_files[0])

# shuffle
random.shuffle(train_files)
random.shuffle(valid_files)

# generate Dataset
train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    args=(train_files,),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 40, 224, 224, 3], [None])
)

valid_dataset = tf.data.Dataset.from_generator(
    data_generator,
    args=(valid_files,),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 40, 224, 224, 3], [None])
)

test_dataset = tf.data.Dataset.from_generator(
    testdata_generator,
    args=(test_files,),
    output_types=(tf.float32),
    output_shapes=([None, 40, 224, 224, 3])
)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=opt, 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
)

# history = model.fit(
#     train_dataset,
#     validation_data=valid_dataset,
#     epochs=30,
# )
# model.save(f"CNN_10_09_2")
model = tf.keras.models.load_model(f"CNN_10_09_2")

# plt.plot(history.history['accuracy'])
# plt.title('c accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.savefig(f"train_img/acc.png")
# plt.close()

# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.savefig(f"train_img/loss.png")
# plt.close()

result = model.predict(
    test_dataset
)
tmp_y_pred = list()
y_pred = list()

for i in result:
    # print(np.argmax(i))
    y_pred.append(np.argmax(i))

# cur = 0
# for test_file in test_files:
#     test_file = test_file.split("/")[-1]
#     frame_cnt = get_mp4_frames(test_file)
#     y_pred.append(most_common(tmp_y_pred[cur: cur+frame_cnt-1]))
#     cur += frame_cnt

print(len(test_files), len(y_pred))
     
df = pd.DataFrame({
    "name": [i.split("/")[-1] for i in test_files],
    "label": y_pred
})
df.to_csv("result_1009_tmp.csv")