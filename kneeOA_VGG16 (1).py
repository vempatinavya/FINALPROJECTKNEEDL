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


# In[24]:


data_path=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train')
print(data_path)


# In[25]:


categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary
print(label_dict)
print(categories)
print(labels)


# In[26]:


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


# In[27]:


data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
label=np.array(label)
from keras.utils import np_utils
new_label=np_utils.to_categorical(label)


# In[29]:


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


# In[30]:


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


# In[31]:


from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# Load the VGG19 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the VGG19 model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the VGG19 model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the data generators for train and test sets
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data
train_data = train_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train'),
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical')

test_data = test_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\test'),
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical')


# In[10]:


history=model.fit(train_data,epochs=20, validation_data=test_data)


# In[18]:


model.save('model.h5')


# In[13]:


from matplotlib import pyplot as plt
# plot the training loss and accuracy
N = 20 #number of epochs
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


#epochs=15
print("Train accuracy:", history.history['accuracy'][-1])
print("Train loss:", history.history['loss'][-1])


# In[13]:


#epochs=20
print("Train accuracy:", history.history['accuracy'][-1])
print("Train loss:", history.history['loss'][-1])


# In[14]:


#epochs=15
print("Test accuracy:", history.history['val_accuracy'][-1])
print("Test loss:", history.history['val_loss'][-1])


# In[15]:


#epochs=20
print("Test accuracy:", history.history['val_accuracy'][-1])
print("Test loss:", history.history['val_loss'][-1])


# In[21]:


from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load an image to classify
img_path = (r'C:\Users\Navya\OneDrive\Desktop\PYTHON PRGMS\Knee X-ray Images\MedicalExpert-III\DoubtfulG1 (66).png')
img = Image.open(img_path).resize((224, 224))
x = np.array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Make a prediction
prediction = model.predict(x)
predicted_class = np.argmax(prediction)
class_names = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']
print("Predicted class:", class_names[predicted_class])


# In[22]:


from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load an image to classify
img_path = (r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\HOD\Knee X-ray Images\MedicalExpert-III\ModerateG3 (62).png')
img = Image.open(img_path).resize((224, 224))
x = np.array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Make a prediction
prediction = model.predict(x)
predicted_class = np.argmax(prediction)
class_names = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5']
print("Predicted class:", class_names[predicted_class])


# In[23]:


from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on test set
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1) # Convert probabilities to class index

# Get true labels
y_true = test_data.classes

# Get class names
class_names = list(test_data.class_indices.keys())

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create a heatmap from the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)

# Add labels and title
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


# In[ ]:




