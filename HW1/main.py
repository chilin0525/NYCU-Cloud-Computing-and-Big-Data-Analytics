from typing import List
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def most_common(lst):
    lst = Counter(lst)
    return lst.most_common(1)[0][0]


def read_mp4(mp4_filename: str)-> List[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4_filename))
    frames = list()

    while cap.isOpened():
        ret, frame = cap.read()
        
        # End of video or error occured
        if not ret:
            break

        # print(ret, frame/255.0, frame.shape, type(frame))
        frame = cv2.resize(frame, (120, 90), interpolation=cv2.INTER_AREA)
        frames.append(frame/255.0)

    cap.release()
    cv2.destroyAllWindows()

    return np.array(frames)


def get_mp4_frames(mp4_filename: str)-> List[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4_filename))
    cnt = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        
        # End of video or error occured
        if not ret:
            break

        cnt +=1 

    cap.release()
    cv2.destroyAllWindows()

    return cnt


def data_generator(filename_list: List[str]):
    """Generator for read mp4 filename from training or testing folder,
    would yield each frame data in video and ground true label(class)
    """
    idx = 0
    while idx<len(filename_list):
        mp4_filename, ground_true = filename_list[idx]
        mp4_filename = mp4_filename.decode('ascii')
        pixel_data = read_mp4(mp4_filename)
        yield (pixel_data, [ground_true]*len(pixel_data))
        idx += 1


def testdata_generator(filename_list: List[str]):
    """Generator for read mp4 filename from training or testing folder,
    would yield each frame data in video and ground true label(class)
    """
    idx = 0
    while idx<len(filename_list):
        mp4_filename = filename_list[idx]
        mp4_filename = mp4_filename.decode('ascii')
        pixel_data = read_mp4(mp4_filename)
        yield pixel_data
        idx += 1


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


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(90, 120, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(39, activation='softmax'))

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

train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    args=(train_files,),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 90, 120, 3], [None])
)

valid_dataset = tf.data.Dataset.from_generator(
    data_generator,
    args=(valid_files,),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 90, 120, 3], [None])
)

print(test_files[0])
test_dataset = tf.data.Dataset.from_generator(
    testdata_generator,
    args=(test_files,),
    output_types=(tf.float32),
    output_shapes=([None, 90, 120, 3])
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
#     epochs=5
# )
# model.save(f"CNN_10_09")
model = tf.keras.models.load_model(f"CNN_10_09")

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
for i in result:
    tmp_y_pred.append(np.argmax(i))

y_pred = list()
cur = 0
for test_file in test_files:
    test_file = test_file.split("/")[-1]
    frame_cnt = get_mp4_frames(test_file)
    y_pred.append(most_common(tmp_y_pred[cur: cur+frame_cnt-1]))
    cur += frame_cnt

print(len(test_files), len(y_pred))
     
df = pd.DataFrame({
    "name": [i.split("/")[-1] for i in test_files],
    "label": y_pred
})
df.to_csv("result.csv")