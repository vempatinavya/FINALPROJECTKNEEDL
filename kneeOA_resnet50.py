#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2,os
import numpy as np
#keras module
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
#matplotlib module
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#sklearn module
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

data_path=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train')
print(data_path)

categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary
print(label_dict)
print(categories)
print(labels)

img_size=224
data=[]
label=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the image  into 224 x 224, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            label.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)
        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image


# In[2]:


data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
label=np.array(label)
from keras.utils import np_utils
new_label=np_utils.to_categorical(label)

import tensorflow as tf

# Resize the images to (224, 224)
x_train_resized = tf.image.resize(data, (224, 224))
x_test_resized = tf.image.resize(data, (224, 224))

# Convert the images to RGB color format with 3 channels
x_train_resized = tf.image.grayscale_to_rgb(x_train_resized)
x_test_resized = tf.image.grayscale_to_rgb(x_test_resized)

# Normalize the pixel values to be in the range [0, 1]
x_train_resized = x_train_resized / 255.0
x_test_resized = x_test_resized / 255.0


# In[3]:


from PIL import Image
import os

# specify the path to the directory containing the images
directory =(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train')

# specify the new size for the images
new_size = (224, 224)

# use a for loop to iterate over all the files in the directory
for filename in os.listdir(directory):
    # check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # open the image using the Image.open() function
        image = Image.open(os.path.join(directory, filename))
        # resize the image
        image = image.resize(new_size)
        # save the resized image
        image.save(os.path.join(directory, filename))


# In[4]:


from PIL import Image
import os

# specify the path to the directory containing the images
directory =(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\test')

# specify the new size for the images
new_size = (224, 224)

# use a for loop to iterate over all the files in the directory
for filename in os.listdir(directory):
    # check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # open the image using the Image.open() function
        image = Image.open(os.path.join(directory, filename))
        # resize the image
        image = image.resize(new_size)
        # save the resized image
        image.save(os.path.join(directory, filename))


# In[6]:


# Import libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directories
train_dir = (r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train')
validation_dir = (r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\test')
#test_dir = (r'C:\Users\Navya\OneDrive\Desktop\PYTHON PRGMS\Knee X-ray Images\MedicalExpert-II')

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(5, activation='sigmoid')(x)

# Create model
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
train_generator = train_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train'),
    
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\test'),
    
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical')





# In[7]:


history = model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=5,validation_data=validation_generator,validation_steps=len(validation_generator))


# In[8]:


#saving the model
model.save('knee_osteoarthritis_detection.h5')


# In[10]:


#evaluate model
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\test'),
    
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')


# In[11]:


print("Train accuracy:", history.history['accuracy'][-1])
print("Train loss:", history.history['loss'][-1])


# In[12]:


print("Test accuracy:", history.history['val_accuracy'][-1])
print("Test loss:", history.history['val_loss'][-1])


# In[13]:


from matplotlib import pyplot as plt
# plot the training loss and accuracy
N = 5 #number of epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")
plt.savefig("CNN_Model")


# In[14]:


from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load an image to classify
img_path = (r'C:\Users\Navya\OneDrive\Desktop\PYTHON PRGMS\Knee X-ray Images\MedicalExpert-III\ModerateG3 (1).png')
img = Image.open(img_path).resize((224, 224))
x = np.array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Make a prediction
prediction = model.predict(x)
predicted_class = np.argmax(prediction)
class_names = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']
print("Predicted class:", class_names[predicted_class])


# In[15]:


from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on test set
y_pred = model.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1) # Convert probabilities to class index

# Get true labels
y_true = validation_generator.classes

# Get class names
class_names = list(validation_generator.class_indices.keys())

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a heatmap from the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)

# Add labels and title
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


# In[ ]:




