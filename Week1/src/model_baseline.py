from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, Model # type: ignore

def build_baseline_model(num_classes, input_shape=(224,224,3), trainable=False):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = trainable
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
