import os
from src.dataset_loader import create_generators
from src.model_baseline import build_baseline_model
from src.train_model import train
from src.evaluate_model import evaluate

DATA_ROOT = "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = "outputs"

def run_week1():
    train_gen, val_gen, test_gen = create_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)
    print("Classes:", train_gen.class_indices)
    model = build_baseline_model(num_classes=len(train_gen.class_indices))
    history, ckpt = train(model, train_gen, val_gen, epochs=12, output_dir=OUTPUT_DIR)
    model.load_weights(ckpt)
    evaluate(model, test_gen, train_gen.class_indices, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    run_week1()
