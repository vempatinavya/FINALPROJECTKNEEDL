#!/usr/bin/env python
# coding: utf-8

# In[2]:



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


# In[3]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD


# In[4]:


# create base model with pre-trained weights from imagenet
base_model = InceptionV3(weights='imagenet', include_top=False)

# add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# define new model with custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all layers in the base model to reuse pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# compile the model with a SGD optimizer and categorical cross-entropy loss
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# define data augmentation parameters
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# load and augment training and validation data
train_generator = train_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\train'),target_size=(224,224),batch_size=64,class_mode='categorical')

val_generator = test_datagen.flow_from_directory(directory=(r'C:\Users\Navya\OneDrive\Desktop\FINAL YEAR PROJECT\archive\test'),
                                                
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')

# train the model on the augmented data
history=model.fit(
        train_generator,
        steps_per_epoch=50 // 32,
        epochs=25,
        validation_data=val_generator,
        validation_steps=50 // 32)

# save the model weights
model.save_weights('inceptionv3_weights.h5')
get_ipython().run_line_magic('history', '')


# In[6]:


from matplotlib import pyplot as plt
# plot the training loss and accuracy
N = 25#number of epochs
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
get_ipython().run_line_magic('history', '')


# In[7]:


print("Train accuracy:", history.history['accuracy'][-1])
print("Train loss:", history.history['loss'][-1])
print("Test accuracy:", history.history['val_accuracy'][-1])
print("Test loss:", history.history['val_loss'][-1])


# In[7]:


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


# In[8]:


from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on test set
y_pred = model.predict(val_generator)
y_pred = np.argmax(y_pred, axis=1) # Convert probabilities to class index

# Get true labels
y_true =val_generator.classes

# Get class names
class_names = list(val_generator.class_indices.keys())

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




