import os
import cv2
import matplotlib.pyplot as plt

dataset_path = "../data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

classes = os.listdir(dataset_path)

print("Toplam sınıf sayısı:", len(classes))

plt.figure(figsize=(12,4))

for i in range(5):

    class_path = os.path.join(dataset_path, classes[i])

    image_name = os.listdir(class_path)[0]

    image_path = os.path.join(class_path, image_name)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title("Class " + classes[i])
    plt.axis("off")

plt.show()