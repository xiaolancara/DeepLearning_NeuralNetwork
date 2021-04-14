import cv2
import tensorflow as tf

CATEGORIES = ["Dog","Cat"]

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model('64x3-CNN.model')

prediction1 = model.predict([prepare('cat_Oreo.jpg')])
prediction2 = model.predict([prepare('cat_Oreo2.jpg')])
prediction3 = model.predict([prepare('dog_Cody.jpg')])
print(CATEGORIES[int(prediction1[0][0])])
print(CATEGORIES[int(prediction2[0][0])])
print(CATEGORIES[int(prediction3[0][0])])

###OUTPUT RESULT###
# Cat
# Cat
# Dog