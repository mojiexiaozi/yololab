# yololab YOLO 🚀, AGPL-3.0 license

from copy import copy

from yololab.models import detect, segment
from yololab.nn.tasks import SegmentationModel
from yololab.utils import DEFAULT_CFG, RANK
from yololab.utils.plotting import plot_images, plot_results


class SemanticTrainer(detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = SegmentationModel(
            cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return segment.SegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(
            file=self.csv, segment=True, on_plot=self.on_plot
        )  # save results.png