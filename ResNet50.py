import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train,y_train),(x_test, y_test)=cifar10.load_data()
x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.

data_augmentation = keras.Sequential(
    [layers.RandomFlip(mode="horizontal"),
     layers.RandomRotation(factor=0.2),
     layers.Normalization(mean=(0.4914, 0.4822, 0.4465), variance=(0.2023, 0.1994, 0.2010))
     ]
)

model_Resnet = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=(32, 32, 3), include_top=False)
model_Resnet.summary()

inputs = layers.Input(shape=(32,32,3))
x = data_augmentation(inputs)
x = model_Resnet(x)
x = layers.Flatten()(x)
x = layers.Dense(units=512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = layers.Dense(units=128, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = layers.Dense(units=64, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
outputs= layers.Dense(units=10)(x)

model=keras.Model( inputs=inputs, outputs=outputs)

#
# save_best = keras.callbacks.ModelCheckpoint("Resnet_checkpoints/",monitor="accuracy",save_best_only= True,save_weights_only=True)
# scheduler = keras.callbacks.ReduceLROnPlateau( monitor= 'loss',factor=0.2, patience=2)
#
# model.compile(
#     optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4),
#     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics = ["accuracy"])
#
# model.fit(x_train, y_train, batch_size= 64, epochs= 10, callbacks=[save_best,scheduler], verbose = 2)
#
# model.evaluate(x_test, y_test, batch_size=64)







