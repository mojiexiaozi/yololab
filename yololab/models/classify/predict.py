# yololab YOLO 🚀, AGPL-3.0 license

import cv2
import torch
from PIL import Image

from yololab.engine.predictor import BasePredictor
from yololab.engine.results import Results
from yololab.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [
                    self.transforms(
                        Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    )
                    for im in img
                ],
                dim=0,
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(
            self.model.device
        )
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            )
        return results
