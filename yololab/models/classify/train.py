# yololab YOLO 🚀, AGPL-3.0 license

import torch
import torchvision

from yololab.data import ClassificationDataset, build_dataloader
from yololab.engine.trainer import BaseTrainer
from yololab.models import classify
from yololab.nn.tasks import ClassificationModel
from yololab.nn.utils import attempt_load_one_weight
from yololab.utils import DEFAULT_CFG, RANK
from yololab.utils.plotting import plot_images, plot_results
from yololab.utils.torch_utils import is_parallel, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        # self.model.names = self.data["names"]
        pass

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ClassificationModel(
            cfg, nc=self.args.nc, verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def setup_model(self):
        # if model is loaded beforehand. No setup needed
        if isinstance(self.model, torch.nn.Module):
            return

        model, ckpt = str(self.model), None
        if model.endswith(".pt"):
            self.model, ckpt = attempt_load_one_weight(model, device="cpu")
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.split(".")[-1] in ("yaml", "yml"):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
        else:
            FileNotFoundError(
                f"ERROR: model={model} not found locally or online. Please check model name."
            )
        ClassificationModel.reshape_outputs(self.model, self.args.nc)

        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        return ClassificationDataset(
            root=img_path, args=self.args, augment=mode == "train", prefix=mode
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # init dataset *.cache only once if DDP
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return classify.ClassificationValidator(
            self.test_loader, self.save_dir, _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        plot_results(
            file=self.csv, classify=True, on_plot=self.on_plot
        )  # save results.png

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(
                -1
            ),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
