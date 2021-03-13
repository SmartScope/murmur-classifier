import os
from cnn import CNN
from classifier import Classifier
from features import FeaturesProcessor

root_dir = "./test_data"
file_set = []

for subdir, dirs, files in os.walk(root_dir):
    for f in files:
        if f.endswith('.wav'):
            file_name = os.path.join(subdir, f)
            file_set.append(file_name[:-4])

print("CNN\n")

cnn = CNN()
for file_name in file_set:
    prediction = cnn.predict(file_name, ensemble=True)
    if prediction is None:
        continue
    print(file_name, prediction)

# print("\nADABOOST\n")

# for file_name in file_set:
#     features_processor = FeaturesProcessor(file_name)
#     features = [features_processor.get_all_features()]

#     classifier = Classifier()
#     prediction = classifier.predict(features, ensemble=True)

#     print(prediction)
