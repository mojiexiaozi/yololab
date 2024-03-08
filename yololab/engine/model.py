# yololab YOLO 🚀, AGPL-3.0 license

import inspect
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

from yololab.cfg import get_cfg, get_save_dir
from yololab.nn.utils import attempt_load_one_weight, guess_model_task
from yololab.nn.tasks import nn, yaml_model_load
from yololab.utils import (
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(nn.Module):
    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.task = task  # task type
        self.model_name = model = str(model).strip()  # strip spaces

        # Check if Triton Server model
        if self.is_triton_model(model):
            self.model = model
            self.task = task
            return

        # Load or create new YOLO model
        model = checks.check_model_file_from_stem(
            model
        )  # add suffix, i.e. yolov8n -> yolov8n.pt
        if Path(model).suffix in (".yaml", ".yml"):
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        self.model_name = model

    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(
            cfg_dict, verbose=verbose and RANK == -1
        )  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {
            **DEFAULT_CFG_DICT,
            **self.overrides,
        }  # combine default and model args (prefer model args)
        self.model.task = self.task

    def _load(self, weights: str, task=None) -> None:
        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task

    def _check_is_pytorch_model(self) -> None:
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = (
            isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        )
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model":
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(
        self, filename: Union[str, Path] = "saved_model.pt", use_dill=True
    ) -> None:
        self._check_is_pytorch_model()
        from yololab import __version__
        from datetime import datetime

        updates = {
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://yololab.com/license)",
            "docs": "https://docs.yololab.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        if not kwargs.get("embed"):
            kwargs["embed"] = [
                len(self.model.model) - 2
            ]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> list:
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (
            sys.argv[0].endswith("yolo") or sys.argv[0].endswith("yololab")
        ) and any(x in sys.argv for x in ("predict", "mode=predict"))

        custom = {"conf": 0.25, "save": is_cli, "mode": "predict"}  # method defaults
        args = {
            **self.overrides,
            **custom,
            **kwargs,
        }  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(
                overrides=args, _callbacks=self.callbacks
            )
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return (
            self.predictor.predict_cli(source=source)
            if is_cli
            else self.predictor(source=source, stream=stream)
        )

    def val(
        self,
        validator=None,
        **kwargs,
    ):
        custom = {"rect": True}  # method defaults
        args = {
            **self.overrides,
            **custom,
            **kwargs,
            "mode": "val",
        }  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(
            args=args, _callbacks=self.callbacks
        )
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):
        self._check_is_pytorch_model()
        from yololab.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {
            **DEFAULT_CFG_DICT,
            **self.model.args,
            **custom,
            **kwargs,
            "mode": "benchmark",
        }
        return benchmark(
            model=self,
            data=kwargs.get(
                "data"
            ),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs,
    ):
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "verbose": False,
        }  # method defaults
        args = {
            **self.overrides,
            **custom,
            **kwargs,
            "mode": "export",
        }  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        nc=80,
        **kwargs,
    ):
        self._check_is_pytorch_model()

        overrides = (
            yaml_load(checks.check_yaml(kwargs["cfg"]))
            if kwargs.get("cfg")
            else self.overrides
        )
        custom = {
            "data": kwargs.get("data") or DEFAULT_CFG_DICT["data"]
        }  # method defaults
        args = {
            **overrides,
            **custom,
            **kwargs,
            "mode": "train",
        }  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        args["nc"] = nc

        self.trainer = (trainer or self._smart_load("trainer"))(
            overrides=args, _callbacks=self.callbacks
        )
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None, cfg=self.model.yaml
            )
            self.model = self.trainer.model

        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = (
                self.trainer.best if self.trainer.best.exists() else self.trainer.last
            )
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(
                self.trainer.validator, "metrics", None
            )  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args,
        **kwargs,
    ):
        self._check_is_pytorch_model()
        if use_ray:
            from yololab.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {
                **self.overrides,
                **custom,
                **kwargs,
                "mode": "train",
            }  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(
                model=self, iterations=iterations
            )

    def _apply(self, fn) -> "Model":
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = (
            self.device
        )  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> list:
        from yololab.nn.autobackend import check_class_names

        return (
            check_class_names(self.model.names)
            if hasattr(self.model, "names")
            else None
        )

    @property
    def device(self) -> torch.device:
        return (
            next(self.model.parameters()).device
            if isinstance(self.model, nn.Module)
            else None
        )

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """Reset arguments when loading a PyTorch model."""
        include = {
            "imgsz",
            "data",
            "task",
            "single_cls",
        }  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(
                    f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet."
                )
            ) from e

    @property
    def task_map(self) -> dict:
        raise NotImplementedError("Please provide task map for your model!")
