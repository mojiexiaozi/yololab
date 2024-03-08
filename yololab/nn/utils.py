# yololab YOLO 🚀, AGPL-3.0 license

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils import model_zoo
import timm

from yololab.utils.torch_utils import intersect_dicts


from yololab.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Conv,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Pose,
    RepC3,
    ResNetLayer,
    Segment,
    WorldDetect,
    RepNCSPELAN4,
    ADown,
    SPPELAN,
    CBFuse,
    CBLinear,
)
from yololab.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    colorstr,
    emojis,
    yaml_load,
)
from yololab.utils.checks import check_requirements, check_suffix, check_yaml
from yololab.utils.torch_utils import make_divisible


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    if not modules:
        modules = {}

    import importlib
    import sys

    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    check_suffix(file=weight, suffix=".pt")
    try:
        with temporary_modules(
            {
                "yololab.yolo.utils": "yololab.utils",
                "yololab.yolo.v8": "yololab.models.yolo",
                "yololab.yolo.data": "yololab.data",
            }
        ):  # for legacy 8.0 Classify and Pose models
            ckpt = torch.load(weight, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        raise e

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, weight  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = (
            {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None
        )  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(
            model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()
        )  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[
        int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))
    ].stride
    assert all(
        ensemble[0].nc == m.nc for m in ensemble
    ), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {
        **DEFAULT_CFG_DICT,
        **(ckpt.get("train_args", {})),
    }  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {
        k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS
    }  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (
        model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()
    )  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(
                f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'."
            )
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    pre_len = 0
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        if "nn." in m:
            m = getattr(nn, m[3:])
        elif "tf_" in m:
            m = m
        else:
            m = globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except:
                    pass

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ADown,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
        ):
            c1, c2 = ch[f], args[0]
            if (
                c2 != nc
            ):  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(
                    min(args[1], max_channels // 2) * width, 8
                )  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                    if args[2] > 1
                    else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in (
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
            ):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        if isinstance(m, str):
            t = m
            m_: nn.Module = timm.create_model(
                m,
                pretrained=False,
                features_only=True,
                exportable=True,
                scriptable=True,
            )
            state_dict = model_zoo.load_url(args[0])
            csd = m_.float().state_dict()
            csd = intersect_dicts(csd, state_dict)  # intersect
            m_.load_state_dict(csd, strict=False)  # load pretrained
            if len(args) > 1:
                m_._stage_out_idx = {s: i for i, s in enumerate(args[1:])}
            preds = m_(torch.randn(1, 3, 224, 224))
            ch = [p.shape[1] for p in preds]
            m_.timm = True
            pre_len = len(ch) - 1

        else:
            m_ = (
                nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            )  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            m_.timm = False
            if i == 0:
                ch = []
            ch.append(c2)

        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + pre_len, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}"
            )  # print
        save.extend(
            x if x >= 0 else m_.i + x
            for x in ([f] if isinstance(f, int) else f)
            if x != -1
        )  # append to savelist
        layers.append(m_)
    print(ch)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(
            f"WARNING ⚠️ yololab YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}."
        )
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(
        r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path)
    )  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(
            1
        )  # n, s, m, l, or x
    return ""


def guess_model_task(model):
    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in ("classify", "classifier", "cls", "fc"):
            return "classify"
        if m == "detect":
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
