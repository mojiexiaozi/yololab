import json
import time
from pathlib import Path
import numpy as np
import torch

from yololab.cfg import get_cfg, get_save_dir
from yololab.nn.autobackend import AutoBackend
from yololab.utils import LOGGER, TQDM, callbacks, colorstr
from yololab.utils.checks import check_imgsz
from yololab.utils.ops import Profile
from yololab.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(
            parents=True, exist_ok=True
        )
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.args.data
            self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (
                trainer.epoch == trainer.epochs - 1
            )
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(
                    f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models"
                )

            if self.device.type in ("cpu", "mps"):
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )

            model.eval()
            model.warmup(
                imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz)
            )  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {
                **stats,
                **trainer.label_loss_items(
                    self.loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(
                    iou >= threshold
                )  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[
                            iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
                        ]
                        matches = matches[
                            np.unique(matches[:, 1], return_index=True)[1]
                        ]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[
                            np.unique(matches[:, 0], return_index=True)[1]
                        ]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        raise NotImplementedError(
            "get_dataloader function not implemented for this validator"
        )

    def build_dataset(self, img_path):
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        return batch

    def postprocess(self, preds):
        return preds

    def init_metrics(self, model):
        pass

    def update_metrics(self, preds, batch):
        pass

    def finalize_metrics(self, *args, **kwargs):
        pass

    def get_stats(self):
        return {}

    def check_stats(self, stats):
        pass

    def print_results(self):
        pass

    def get_desc(self):
        pass

    @property
    def metric_keys(self):
        return []

    def on_plot(self, name, data=None):
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        pass

    def plot_predictions(self, batch, preds, ni):
        pass

    def pred_to_json(self, preds, batch):
        pass

    def eval_json(self, stats):
        pass
