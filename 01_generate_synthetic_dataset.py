import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm
import json

CONFIG = {
    'gt_csvs': ['HMPD-Gen/gt.csv', 'HMPD-Gen/gtPossible.csv'],
    'images_dir': 'HMPD-Gen/images',
    'channel': '_P',
    'output_dir': 'microplastic_data',
    'num_images': 2000,
    'image_size': (640, 640),
    'particles_per_image': (1, 8),
    'negative_ratio': 0.5,
    'min_patch_size': 32,
    'min_background_size': 200,
    'random_seed': 42
}

class SyntheticDatasetGenerator:
    def __init__(self, config):
        self.config = config
        random.seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        
        if 'gt_csvs' in config:
            self.gt_paths = [Path(csv_path) for csv_path in config['gt_csvs']]
        elif 'gt_csv' in config:
            self.gt_paths = [Path(config['gt_csv'])]
        else:
            raise ValueError("Config must contain either 'gt_csv' or 'gt_csvs'")
            
        self.images_dir = Path(config['images_dir'])
        self.output_dir = Path(config['output_dir'])
        self.images_output = self.output_dir / 'images'
        self.labels_output = self.output_dir / 'labels'
        self.images_output.mkdir(parents=True, exist_ok=True)
        self.labels_output.mkdir(parents=True, exist_ok=True)
        
        self.positive_patches = self._load_positive_patches()
        self.background_patches = self._load_background_patches()
    
    def _load_positive_patches(self):
        all_positive = []
        for gt_path in self.gt_paths:
            print(f"Loading: {gt_path}")
            if not gt_path.exists():
                raise FileNotFoundError(f"GT file not found: {gt_path}")
            df = pd.read_csv(gt_path)
            positive = df[df['classes'] == 1]['patchids'].tolist()
            all_positive.extend(positive)
            print(f"  Found {len(positive)} positive patches")
        print(f"Total {len(all_positive)} positive patches")
        return all_positive
    
    def _load_background_patches(self):
        all_negative = []
        for gt_path in self.gt_paths:
            df = pd.read_csv(gt_path)
            negative = df[df['classes'] == 0]['patchids'].tolist()
            all_negative.extend(negative)
            print(f"  Found {len(negative)} negative patches")
        print(f"Total {len(all_negative)} negative patches")
        
        valid_backgrounds = []
        for patch_id in all_negative:
            patch = self._load_patch(patch_id)
            if patch is not None:
                h, w = patch.shape[:2]
                if h >= self.config['min_background_size'] and w >= self.config['min_background_size']:
                    valid_backgrounds.append(patch_id)
        print(f"Valid backgrounds: {len(valid_backgrounds)}")
        if len(valid_backgrounds) == 0:
            raise ValueError("No valid background patches")
        return valid_backgrounds
    
    def _load_patch(self, patch_id):
        patch_path = self.images_dir / f"{patch_id}{self.config['channel']}.bmp"
        if not patch_path.exists():
            return None
        img = cv2.imread(str(patch_path))
        if img is None or img.shape[0] < self.config['min_patch_size'] or img.shape[1] < self.config['min_patch_size']:
            return None
        return img
    
    def _normalize_intensity(self, patch, target_mean=120, target_std=15):
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        current_mean = np.mean(patch_gray)
        current_std = np.std(patch_gray)
        if current_std < 1:
            current_std = 1
        normalized = (patch.astype(np.float32) - current_mean) * (target_std / current_std) + target_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return normalized
    
    def _create_realistic_canvas(self):
        target_h, target_w = self.config['image_size']
        bg_patch_id = random.choice(self.background_patches)
        bg_patch = self._load_patch(bg_patch_id)
        
        if bg_patch is None:
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 120
        else:
            bg_h, bg_w = bg_patch.shape[:2]
            bg_patch = self._normalize_intensity(bg_patch)
            
            if bg_h >= target_h and bg_w >= target_w:
                start_y = (bg_h - target_h) // 2
                start_x = (bg_w - target_w) // 2
                canvas = bg_patch[start_y:start_y+target_h, start_x:start_x+target_w].copy()
            else:
                canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
                weight_map = np.zeros((target_h, target_w), dtype=np.float32)
                
                for y in range(0, target_h, bg_h // 2):
                    for x in range(0, target_w, bg_w // 2):
                        y_end = min(y + bg_h, target_h)
                        x_end = min(x + bg_w, target_w)
                        patch_h = y_end - y
                        patch_w = x_end - x
                        
                        rand_bg_id = random.choice(self.background_patches)
                        rand_bg = self._load_patch(rand_bg_id)
                        
                        if rand_bg is not None:
                            rand_bg = self._normalize_intensity(rand_bg)
                            rand_bg_h, rand_bg_w = rand_bg.shape[:2]
                            crop_h = min(patch_h, rand_bg_h)
                            crop_w = min(patch_w, rand_bg_w)
                            crop_y = (rand_bg_h - crop_h) // 2
                            crop_x = (rand_bg_w - crop_w) // 2
                            cropped = rand_bg[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                            weight = np.ones((crop_h, crop_w), dtype=np.float32)
                            canvas[y:y+crop_h, x:x+crop_w] += cropped.astype(np.float32) * weight[:, :, np.newaxis]
                            weight_map[y:y+crop_h, x:x+crop_w] += weight
                
                weight_map[weight_map == 0] = 1
                canvas = (canvas / weight_map[:, :, np.newaxis]).astype(np.uint8)
        
        canvas = cv2.GaussianBlur(canvas, (5, 5), 1.0)
        return canvas
    
    def _place_patch(self, canvas, patch, placed_regions, attempts=50):
        canvas_h, canvas_w = canvas.shape[:2]
        patch_h, patch_w = patch.shape[:2]
        patch = self._normalize_intensity(patch)
        
        for _ in range(attempts):
            x = random.randint(0, canvas_w - patch_w)
            y = random.randint(0, canvas_h - patch_h)
            overlap = False
            for (px, py, pw, ph) in placed_regions:
                if not (x + patch_w < px or x > px + pw or y + patch_h < py or y > py + ph):
                    overlap = True
                    break
            
            if not overlap:
                mask = np.ones((patch_h, patch_w), dtype=np.float32)
                border = 5
                for i in range(border):
                    alpha = (i + 1) / border
                    mask[i, :] *= alpha
                    mask[-i-1, :] *= alpha
                    mask[:, i] *= alpha
                    mask[:, -i-1] *= alpha
                
                roi = canvas[y:y+patch_h, x:x+patch_w].astype(np.float32)
                blended = (patch.astype(np.float32) * mask[:, :, np.newaxis] + roi * (1 - mask[:, :, np.newaxis]))
                canvas[y:y+patch_h, x:x+patch_w] = blended.astype(np.uint8)
                
                x_center = (x + patch_w / 2) / canvas_w
                y_center = (y + patch_h / 2) / canvas_h
                width = patch_w / canvas_w
                height = patch_h / canvas_h
                bbox = (0, x_center, y_center, width, height)
                region = (x, y, patch_w, patch_h)
                return True, bbox, region
        
        return False, None, None
    
    def generate_image(self, img_id):
        canvas = self._create_realistic_canvas()
        bboxes = []
        placed_regions = []
        is_negative = random.random() < self.config.get('negative_ratio', 0.5)
        
        if is_negative:
            num_particles = 0
        else:
            num_particles = random.randint(*self.config['particles_per_image'])
            if num_particles == 0:
                num_particles = 1
        
        if num_particles > 0:
            available_patches = self.positive_patches.copy()
            random.shuffle(available_patches)
            placed = 0
            attempts_per_particle = 100
            
            for patch_id in available_patches:
                if placed >= num_particles:
                    break
                patch = self._load_patch(patch_id)
                if patch is None:
                    continue
                success, bbox, region = self._place_patch(canvas, patch, placed_regions, attempts=attempts_per_particle)
                if success:
                    bboxes.append(bbox)
                    placed_regions.append(region)
                    placed += 1
            
            if placed == 0 and num_particles > 0:
                for patch_id in available_patches[:20]:
                    patch = self._load_patch(patch_id)
                    if patch is None:
                        continue
                    success, bbox, region = self._place_patch(canvas, patch, placed_regions, attempts=200)
                    if success:
                        bboxes.append(bbox)
                        placed_regions.append(region)
                        placed += 1
                        break
        
        return canvas, bboxes, len(bboxes)
    
    def save_image_and_label(self, img_id, canvas, bboxes):
        img_path = self.images_output / f"microplastic_synthetic_{img_id:05d}.jpg"
        cv2.imwrite(str(img_path), canvas)
        label_path = self.labels_output / f"microplastic_synthetic_{img_id:05d}.txt"
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                class_id, x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def generate_dataset(self):
        print("=" * 60)
        print(f"Generating {self.config['num_images']} synthetic images")
        print("=" * 60)
        total_particles = 0
        negative_count = 0
        
        with tqdm(total=self.config['num_images'], desc="Generating") as pbar:
            for i in range(self.config['num_images']):
                canvas, bboxes, placed = self.generate_image(i)
                self.save_image_and_label(i, canvas, bboxes)
                total_particles += placed
                if len(bboxes) == 0:
                    negative_count += 1
                pbar.update(1)
                pbar.set_postfix({'particles': placed, 'neg': negative_count})
        
        info = {
            'num_images': self.config['num_images'],
            'positive_images': self.config['num_images'] - negative_count,
            'negative_images': negative_count,
            'total_particles': total_particles,
            'avg_particles_per_image': total_particles / self.config['num_images'],
            'image_size': self.config['image_size'],
            'config': self.config
        }
        
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print("=" * 60)
        print(f"Complete: {self.output_dir.absolute()}")
        print(f"Positive: {self.config['num_images'] - negative_count} | Negative: {negative_count}")
        print(f"Particles: {total_particles} ({total_particles / self.config['num_images']:.1f} avg/image)")
        print("=" * 60)
        return info

def main():
    generator = SyntheticDatasetGenerator(CONFIG)
    generator.generate_dataset()

if __name__ == "__main__":
    main()
