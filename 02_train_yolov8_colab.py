"""
YOLOv8 Training Script for Microplastic Detection

Trains YOLOv8 on synthetic dataset. Optimized for Google Colab GPU.
Handles dataset preparation, model training, and evaluation.
"""

print("=" * 70)
print("YOLOv8 Training")
print("=" * 70)

print("\nInstalling dependencies...")
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
from datetime import datetime

CONFIG = {
    'dataset_dir': 'microplastic_data',
    'model': 'yolov8n.pt',
    'epochs': 50,
    'batch_size': 16,
    'img_size': 640,
    'patience': 10,
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'lrf': 0.01,
    'project': 'yolov8_microplastic',
    'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
}

def prepare_yolo_dataset(dataset_dir):
    """Prepare and split dataset into train/val/test sets."""
    print("\n" + "=" * 70)
    print("Preparing dataset")
    print("=" * 70)
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Dataset structure incorrect. Need {dataset_dir}/images/ and {dataset_dir}/labels/")
    
    images = list(images_dir.glob('*.jpg'))
    labels = list(labels_dir.glob('*.txt'))
    
    print(f"Found {len(images)} images, {len(labels)} labels")
    
    import random
    random.seed(42)
    
    image_files = [img.stem for img in images]
    random.shuffle(image_files)
    
    train_split = int(len(image_files) * 0.7)
    val_split = int(len(image_files) * 0.9)
    
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_images = dataset_path / split_name / 'images'
        split_labels = dataset_path / split_name / 'labels'
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)
        
        for filename in split_files:
            src_img = images_dir / f"{filename}.jpg"
            dst_img = split_images / f"{filename}.jpg"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            src_lbl = labels_dir / f"{filename}.txt"
            dst_lbl = split_labels / f"{filename}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
    
    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['microplastic']
    }
    
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"Dataset prepared: {yaml_path}")
    return str(yaml_path)

def train_yolo():
    """Train YOLOv8 model."""
    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70)
    
    data_yaml = prepare_yolo_dataset(CONFIG['dataset_dir'])
    
    print(f"\nLoading model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    
    print("\nTraining configuration:")
    for key, value in CONFIG.items():
        if key != 'dataset_dir':
            print(f"  {key}: {value}")
    
    print("\nTraining started (30-60 minutes)...")
    print("-" * 70)
    
    results = model.train(
        data=data_yaml,
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch_size'],
        imgsz=CONFIG['img_size'],
        patience=CONFIG['patience'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        verbose=True,
        plots=True,
        save=True
    )
    
    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)
    
    # YOLOv8 saves to runs/detect/{project}/{name} by default
    save_dir = Path('runs/detect') / CONFIG['project'] / CONFIG['name']
    best_model = save_dir / 'weights' / 'best.pt'
    
    print(f"Results: {save_dir}")
    print(f"Best model: {best_model}")
    
    return model, best_model

def evaluate_model(model_path):
    """Evaluate model on test set."""
    print("\n" + "=" * 70)
    print("Evaluating model")
    print("=" * 70)
    
    model = YOLO(model_path)
    data_yaml = Path(CONFIG['dataset_dir']) / 'data.yaml'
    metrics = model.val(data=str(data_yaml), split='test')
    
    print("\nTest results:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.p[0]:.3f}")
    print(f"  Recall: {metrics.box.r[0]:.3f}")
    
    return metrics

if __name__ == "__main__":
    try:
        model, best_model_path = train_yolo()
        metrics = evaluate_model(best_model_path)
        
        print("\n" + "=" * 70)
        print("Training complete")
        print(f"Model: {best_model_path}")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise
