from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def create_generators(train_dir, val_dir, test_dir, image_size=(224,224), batch_size=32):
    train_aug = ImageDataGenerator(rescale=1./255)
    val_test = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')
    val_gen   = val_test.flow_from_directory(val_dir,   target_size=image_size, batch_size=batch_size, class_mode='categorical')
    test_gen  = val_test.flow_from_directory(test_dir,  target_size=image_size, batch_size=batch_size, class_mode='categorical')
    return train_gen, val_gen, test_gen
