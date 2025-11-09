import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
import matplotlib.pyplot as plt

def train(model, train_gen, val_gen, epochs=12, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "best_model.h5")
    cp = ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max")
    es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[cp, es])

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(output_dir,"accuracy_plot.png"))
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(output_dir,"loss_plot.png"))
    return history, ckpt_path
