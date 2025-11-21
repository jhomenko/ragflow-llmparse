import os
from typing import List, Dict, Union

from doclayout_yolo import YOLOv10
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
try:
    # Ultralytics YOLO supports OpenVINO-exported folders; optional.
    from ultralytics import YOLO as UltralyticsYOLO
except Exception:
    UltralyticsYOLO = None

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class DocLayoutYOLOModel:
    def __init__(
        self,
        weight: str,
        device: str = "cuda",
        imgsz: int = 1280,
        conf: float = 0.1,
        iou: float = 0.45,
        use_openvino: bool = False,
        ov_device: str = None,
    ):
        self.is_openvino = use_openvino
        # Prefer Ultralytics loader for OpenVINO exports; fallback to original loader.
        if self.is_openvino and UltralyticsYOLO is not None:
            self.model = UltralyticsYOLO(weight)
            self.ov_device = ov_device or device
        else:
            self.is_openvino = False  # Ensure consistency if fallback
            self.model = YOLOv10(weight).to(device)
            self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def _parse_prediction(self, prediction) -> List[Dict]:
        layout_res = []

        # 容错处理
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return layout_res

        def _to_cpu_array(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            return np.asarray(x)

        xyxy_arr = _to_cpu_array(prediction.boxes.xyxy)
        conf_arr = _to_cpu_array(prediction.boxes.conf)
        cls_arr = _to_cpu_array(prediction.boxes.cls)

        for xyxy, conf, cls in zip(xyxy_arr, conf_arr, cls_arr):
            xmin, ymin, xmax, ymax = list(map(int, xyxy.tolist()))
            layout_res.append(
                {
                    "category_id": int(cls.item() if hasattr(cls, "item") else cls),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item() if hasattr(conf, "item") else conf), 3),
                }
            )
        return layout_res

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        predict_kwargs = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }
        if self.is_openvino and getattr(self, "ov_device", None):
            predict_kwargs["device"] = self.ov_device
        prediction = self.model.predict(image, **predict_kwargs)[0]
        return self._parse_prediction(prediction)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4
    ) -> List[List[Dict]]:
        results = []
        predict_kwargs = {
            "imgsz": self.imgsz,
            "iou": self.iou,
            "verbose": False,
        }
        if self.is_openvino and getattr(self, "ov_device", None):
            predict_kwargs["device"] = self.ov_device
        with tqdm(total=len(images), desc="Layout Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx: idx + batch_size]
                if batch_size == 1:
                    conf = 0.9 * self.conf
                else:
                    conf = self.conf
                predictions = self.model.predict(
                    batch,
                    conf=conf,
                    **predict_kwargs,
                )
                for pred in predictions:
                    results.append(self._parse_prediction(pred))
                pbar.update(len(batch))
        return results

    def visualize(
            self,
            image: Union[np.ndarray, Image.Image],
            results: List
    ) -> Image.Image:

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)
        for res in results:
            poly = res['poly']
            xmin, ymin, xmax, ymax = poly[0], poly[1], poly[4], poly[5]
            print(
                f"Detected box: {xmin}, {ymin}, {xmax}, {ymax}, Category ID: {res['category_id']}, Score: {res['score']}")
            # 使用PIL在图像上画框
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            # 在框旁边画置信度
            draw.text((xmax + 10, ymin + 10), f"{res['score']:.2f}", fill="red", font_size=22)
        return image


if __name__ == '__main__':
    image_path = r"C:\Users\zhaoxiaomeng\Downloads\下载1.jpg"
    doclayout_yolo_weights = os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
    device = 'cuda'
    model = DocLayoutYOLOModel(
        weight=doclayout_yolo_weights,
        device=device,
    )
    image = Image.open(image_path)
    results = model.predict(image)

    image = model.visualize(image, results)

    image.show()  # 显示图像
