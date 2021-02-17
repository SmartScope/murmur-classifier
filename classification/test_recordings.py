import os
from cnn import CNN

root_dir = "./test_data"
file_set = []

for subdir, dirs, files in os.walk(root_dir):
    for f in files:
        if f.endswith('.wav'):
            file_name = os.path.join(subdir, f)
            file_set.append(file_name[:-4])

for file_name in file_set:
    cnn = CNN()
    prediction = cnn.predict(file_name, debug_mode=True)
    if prediction is None:
        continue
    print(file_name)