import os
import sys
import cv2
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
from ultralytics import YOLO

sys.path.append(os.path.abspath("LiteMono"))
import networks
from layers import disp_to_depth
from options import LiteMonoOptions


class RiskDetector:
    def __init__(self, yolo_model_path, encoder_path, decoder_path, gt_depth_path,
                 labels_json, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # Load Encoder and Decoder
        encoder_dict = torch.load(encoder_path, map_location=self.device)
        decoder_dict = torch.load(decoder_path, map_location=self.device)
        self.feed_width = encoder_dict['width']
        self.feed_height = encoder_dict['height']

        self.encoder = networks.LiteMono(model="lite-mono", height=self.feed_height, width=self.feed_width)
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in self.encoder.state_dict()})
        self.encoder.to(self.device).eval()

        self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc, scales=range(3))
        self.depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in self.depth_decoder.state_dict()})
        self.depth_decoder.to(self.device).eval()

        self.gt_depths = np.load(gt_depth_path, allow_pickle=True)["data"]
        with open(labels_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.labels = [item['name'] for item in data['categories'] if 'name' in item]

        self.risk_baseline = {
            'person': 30.0, 'car': 50.0, 'rider': 40.0, 'bus': 50.0,
            'truck': 50.0, 'bike': 40.0, 'motor': 40.0,
            'traffic light': 20.0, 'traffic sign': 20.0
        }

    def detect_objects(self, img):
        results = self.yolo_model(img)
        bbox_tensor = results[0].boxes.xyxy.cpu().numpy()
        cls_tensor = results[0].boxes.cls.cpu().numpy()
        return bbox_tensor, cls_tensor

    def estimate_depth(self, img, original_width, original_height):
        input_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')
        input_image = input_image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
        input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(input_tensor)
            outputs = self.depth_decoder(features)
            disp = outputs["disp", 0]
            scaled_disp, _ = disp_to_depth(disp, 1e-3, 80)
            pred_disp = scaled_disp.squeeze().cpu().numpy()

            scale_ratios = []
            for gt_depth in self.gt_depths:
                gt_h, gt_w = gt_depth.shape[:2]
                pred_disp_resized = cv2.resize(pred_disp, (gt_w, gt_h))
                pred_depth_temp = 1.0 / pred_disp_resized
                mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80)
                if np.count_nonzero(mask) == 0:
                    continue
                ratio = np.median(gt_depth[mask]) / np.median(pred_depth_temp[mask])
                scale_ratios.append(ratio)

            global_scale = np.median(scale_ratios)
            pred_depth = 1.0 / cv2.resize(pred_disp, (original_width, original_height))
            pred_depth *= global_scale

        return pred_depth

    def calculate_risk_score_frame(self, frame):
        original_height, original_width = frame.shape[:2]
        bbox_tensor, cls_tensor = self.detect_objects(frame)
        if bbox_tensor is None or cls_tensor is None or len(bbox_tensor) == 0:
            return None

        pred_depth = self.estimate_depth(frame, original_width, original_height)
        depth_map_resized = cv2.resize(pred_depth, (original_width, original_height))

        objects_info = []
        for i, bbox in enumerate(bbox_tensor):
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_width, x2), min(original_height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            depth_crop = depth_map_resized[y1:y2, x1:x2]
            if depth_crop.size == 0:
                continue

            min_val = float(np.min(depth_crop))
            center_x = (x1 + x2) / 2.0

            location = 'left' if center_x < original_width / 3 else 'center' if center_x < 2 * original_width / 3 else 'right'
            class_idx = cls_tensor[i]
            class_name = self.labels[int(class_idx)] if class_idx < len(self.labels) else "others"
            baseline = self.risk_baseline.get(class_name, 1.0)
            risk_score = baseline * (1.0 / (min_val + 1e-3))

            objects_info.append((class_name, location, risk_score, min_val, (x1, y1, x2, y2)))

        return max(objects_info, key=lambda obj: obj[2]) if objects_info else None

    @staticmethod
    def draw_max_risk_bbox(frame, best_obj):
        class_name, location, risk_score, min_depth, bbox = best_obj
        x1, y1, x2, y2 = bbox
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} | {location} | Risk: {risk_score:.2f} | MinD: {min_depth:.2f}"
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return display_frame


def run_video_analysis(video_path, save_path=None):
    detector = RiskDetector(
        yolo_model_path="./models/best.pt",
        encoder_path="./models/lite-mono_640x192/encoder.pth",
        decoder_path="./models/lite-mono_640x192/depth.pth",
        gt_depth_path="./gt_depths.npz",
        labels_json="bdd100k_labels_images_train_coco.json"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        best_obj = detector.calculate_risk_score_frame(frame)
        if best_obj:
            frame = detector.draw_max_risk_bbox(frame, best_obj)

        if writer:
            writer.write(frame)
        else:
            cv2.imshow('Real-Time Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    input_dir = '../test'
    output_dir = '../out'
    for file in os.listdir(input_dir):
        print('from ', os.path.join(input_dir, file))
        print('to', os.path.join(output_dir, file))
        run_video_analysis(os.path.join(input_dir, file),os.path.join(output_dir, file))
        print(f'file {file} done')