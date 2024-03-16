# yololab YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from yololab.engine.model import Model
from yololab.models import classify, detect, segment, pose, obb
from yololab.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
)
from yololab.utils import yaml_load, ROOT


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {
            ".pt",
            ".yaml",
            ".yml",
        }:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": classify.ClassificationTrainer,
                "validator": classify.ClassificationValidator,
                "predictor": classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": detect.DetectionTrainer,
                "validator": detect.DetectionValidator,
                "predictor": detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": segment.SegmentationTrainer,
                "validator": segment.SegmentationValidator,
                "predictor": segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": pose.PoseTrainer,
                "validator": pose.PoseValidator,
                "predictor": pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": obb.OBBTrainer,
                "validator": obb.OBBValidator,
                "predictor": obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt") -> None:
        super().__init__(model=model, task="detect")

        # Assign default COCO class names
        self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": detect.DetectionValidator,
                "predictor": detect.DetectionPredictor,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
