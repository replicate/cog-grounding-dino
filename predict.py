from cog import BasePredictor, BaseModel, Input, Path
import os
from typing import Optional, List
import torch
from cv2 import imwrite as cv2_imwrite
import file_utils
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate


WEIGHTS_CACHE_DIR = "/src/weights"
HUGGINGFACE_CACHE_DIR = "/src/hf-cache/"
os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = HUGGINGFACE_CACHE_DIR
file_utils.download_grounding_dino_weights(
    grounding_dino_weights_dir=WEIGHTS_CACHE_DIR,
    hf_cache_dir=HUGGINGFACE_CACHE_DIR,
)


class ModelOutput(BaseModel):
    detections: List
    result_image: Optional[Path]


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_model(
            "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"{WEIGHTS_CACHE_DIR}/groundingdino_swint_ogc.pth",
            device=self.device,
        )

    def predict(
        self,
        image: Path = Input(description="Input image to query", default=None),
        query: str = Input(
            description="Comma seperated names of the objects to be detected in the image",
            default=None,
        ),
        box_threshold: float = Input(
            description="Confidence level for object detection",
            ge=0,
            le=1,
            default=0.25,
        ),
        text_threshold: float = Input(
            description="Confidence level for object detection",
            ge=0,
            le=1,
            default=0.25,
        ),
        show_visualisation: bool = Input(
            description="Draw and visualize bounding boxes on the image", default=True
        ),
    ) -> ModelOutput:
        image_source, image = load_image(image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # Convert boxes from center, width, height to top left, bottom right
        height, width, _ = image_source.shape
        boxes_original_size = boxes * torch.Tensor([width, height, width, height])
        xyxy = (
            box_convert(boxes=boxes_original_size, in_fmt="cxcywh", out_fmt="xyxy")
            .numpy()
            .astype(int)
        )

        # Prepare the output
        detections = []
        for box, score, label in zip(xyxy, logits, phrases):
            data = {
                "label": label,
                "confidence": score.item(),  # torch tensor to float
                "bbox": box,
            }
            detections.append(data)

        # Visualize the output if requested
        result_image_path = None
        if show_visualisation:
            result_image_path = "/tmp/result.png"
            result_image = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases,
            )
            cv2_imwrite(result_image_path, result_image)

        return ModelOutput(
            detections=detections,
            result_image=Path(result_image_path) if show_visualisation else None,
        )
