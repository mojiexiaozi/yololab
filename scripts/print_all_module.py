import yaml
from glob import glob

use = {
    "Detect",
    "Segment",
    "WorldDetect",
    "Classify",
    "CBFuse",
    "C3Ghost",
    "Silence",
    "C2",
    "Conv",
    "nn.Upsample",
    "RTDETRDecoder",
    "C2f",
    "Pose",
    "ADown",
    "SPPELAN",
    "ResNetLayer",
    "Concat",
    "C2fAttn",
    "SPPF",
    "GhostConv",
    "CBLinear",
    "RepNCSPELAN4",
    "OBB",
    "ImagePoolingAttn",
}
if __name__ == "__main__":
    modules = set()
    for cfg in glob("**/models/**/*.yaml", recursive=True):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)
            for f, repeats, module, args in cfg["head"] + cfg["backbone"]:
                modules.add(module)
    print(modules)
