"""Thin adapters that compute the NATIVE score of each FIQA method on aligned 112x112 images, using the model code and
weights of the official repositories (cloned by the user, see README). Nothing from these repositories is copied here.

Each loader returns a function  score(paths) -> np.ndarray  of native scores. All images are read with OpenCV, resized
to the input size of the method, and converted to RGB, as in the paper; method-specific normalization follows the
official implementations."""
import os
import sys
from pathlib import Path
import cv2
import numpy as np


def _rgb(paths, size=112):
    """cv2.imread (BGR) -> resize (bilinear) -> RGB, as uint8 NHWC."""
    out = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        out.append(cv2.cvtColor(cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def _tensor(paths, scale="pm1"):
    """RGB batch as float CUDA tensor NCHW in [-1, 1] ("pm1") or [0, 1] ("unit")."""
    import torch
    x = torch.from_numpy(_rgb(paths)).permute(0, 3, 1, 2).contiguous().float().cuda() / 255.0
    return (x - 0.5) / 0.5 if scale == "pm1" else x


def _batched(fn, bs):
    import torch
    def run(paths):
        with torch.inference_mode():
            return np.concatenate([fn(paths[i:i + bs]) for i in range(0, len(paths), bs)])
    return run


def _import(repo, *sub):
    """Put the official repository (or sub-folders of it) on the Python path, last one first."""
    for s in sub or ("",):
        sys.path.insert(0, str(Path(repo) / s))


# ---------------------------------------------------------------- supervised
def faceqnet(repo, weights):
    """FaceQnet v1 (https://github.com/uam-biometrics/FaceQnet), weights FaceQnet_v1.h5. RGB 224x224, no normalization."""
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    model = tf.keras.models.load_model(weights, compile=False)
    def run(paths):
        return np.concatenate([model(_rgb(paths[i:i + 64], 224).astype(np.float32), training=False).numpy().reshape(-1)
                               for i in range(0, len(paths), 64)])
    return run


def sdd_fiqa(repo, weights):
    """SDD-FIQA (https://github.com/Tencent/TFace, folder quality/), weights SDD_FIQA_checkpoints_r50.pth."""
    import torch
    _import(repo, "quality")
    from generate_pseudo_labels.extract_embedding.model import model as sdd
    net = sdd.R50([112, 112], use_type="Qua").cuda()
    sd = net.state_dict(); sd.update({k.replace("module.", ""): v for k, v in torch.load(weights, map_location="cpu").items()})
    net.load_state_dict(sd); net.eval()
    return _batched(lambda p: net(_tensor(p)).reshape(-1).cpu().numpy(), 128)


def clib_fiqa(repo, weights, clip_rn50):
    """CLIB-FIQA (https://github.com/oufuzhao/CLIB-FIQA), weights CLIB-FIQA_R50.pth and the OpenAI CLIP RN50.pt."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from itertools import product
    _import(repo)
    from model import clip
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model, _ = clip.load(clip_rn50, device="cuda", jit=False)
    sd, pre = model.state_dict(), torch.load(weights, map_location="cuda")
    sd.update({k.replace("module.", ""): v for k, v in pre.items() if k.replace("module.", "") in sd})
    model.load_state_dict(sd); model.eval()
    q = ["bad", "poor", "fair", "good", "perfect"]
    lists = [["hazy", "blurry", "clear"], ["obstructed", "unobstructed"], ["profile", "slight angle", "frontal"],
             ["exaggerated expression", "typical expression"], ["extreme lighting", "normal lighting"], q]
    texts = torch.cat([clip.tokenize(f"a photo of a {b}, {o}, and {p} face with {e} under {l}, which is of {qq} quality")
                       for b, o, p, e, l, qq in product(*lists)]).cuda()
    tf = T.Compose([T.ToPILImage(), T.Resize([224, 224]), T.ToTensor(),
                    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    anchors = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).cuda()
    def fn(paths):
        ims = [cv2.imread(str(p)) for p in paths]
        x = torch.stack([tf(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in ims]).cuda()
        logits, _ = model.forward(x, texts)
        prob = F.softmax(logits.view(len(paths), -1), dim=1).view(-1, 3, 2, 3, 2, 2, 5)
        return (prob.sum(1).sum(1).sum(1).sum(1).sum(1) * anchors).sum(1).cpu().numpy()
    return _batched(fn, 64)


# ---------------------------------------------------------------- unsupervised, non-FR-model
def diffiqa_r(repo, weights):
    """DifFIQA(R) (https://github.com/LSIbabnikz/DifFIQA, folder diffiqa(r)/), weights diffiqa(r).pth."""
    import torch
    from collections import OrderedDict
    _import(repo, "diffiqa(r)")
    from model_weights.iresnet import iresnet100
    net = torch.nn.Sequential(OrderedDict([("base", iresnet100()), ("linear", torch.nn.Linear(512, 1))]))
    net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True); net = net.cuda().eval()
    return _batched(lambda p: net(_tensor(p)).reshape(-1).cpu().numpy(), 128)


def ediffiqa_l(repo, weights):
    """eDifFIQA(L) (https://github.com/LSIbabnikz/eDifFIQA), weights ediffiqaL.pth."""
    import torch
    _import(repo)
    from model.iresnet import iresnet100
    from model.mlp import MLP
    from model.ediffiqa import eDifFIQA
    net = eDifFIQA(backbone_model=iresnet100(), quality_head=MLP(in_dim=512, hidden_dim=1024, out_dim=1), return_feat=False)
    net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True); net = net.cuda().eval()
    return _batched(lambda p: net(_tensor(p)).reshape(-1).cpu().numpy(), 128)


# ---------------------------------------------------------------- unsupervised, FR-probing
def ser_fiq(repo, weights=None, T=100, gpu=0):
    """SER-FIQ (https://github.com/pterhoer/FaceImageQuality, MXNet, insightface model of the repo). Stochastic:
    T dropout passes per image, raw score 2 / (1 + exp(mean pairwise distance)). Seeds as in the paper."""
    import mxnet as mx
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import euclidean_distances
    os.chdir(repo); _import(repo)
    from face_image_quality import SER_FIQ
    np.random.seed(0); mx.random.seed(0)
    ser = SER_FIQ(gpu=None if gpu < 0 else gpu)
    def run(paths):
        out = []
        for p in paths:
            img = np.transpose(_rgb([p])[0], (2, 0, 1))
            X = normalize(ser.insightface(mx.nd.array(np.repeat(img[None], T, axis=0), ctx=ser.device)).asnumpy(), axis=1)
            out.append(2.0 * (1.0 / (1.0 + np.exp(float(np.mean(euclidean_distances(X, X)[np.triu_indices(T, k=1)]))))))
        return np.array(out)
    return run


def grafiqs_l(repo, weights):
    """GraFIQs(L) (https://github.com/jankolf/GraFIQs), weights resnet100_ms1mv2_arcface.pth. Native score: sum of the
    absolute gradients of the BN-statistics loss w.r.t. block 2 (higher = more degradation, pi_m = -1)."""
    import torch
    import torch.autograd as autograd
    _import(repo)
    from extract_grafiqs import get_model
    model = get_model(nn_architecture="iresnet100", rank=torch.device("cuda:0"), nn_weights_path=str(weights))
    def run(paths):
        out = []
        for p in paths:
            x = _tensor([p]).requires_grad_(True)
            bn_score, (_, _, block2, _, _, _) = model.get_BN(x)
            g = autograd.grad(outputs=bn_score, inputs=block2, retain_graph=False, create_graph=False)[0]
            out.append(float(torch.abs(g[0]).sum().cpu()))
        return np.array(out)
    return run


def froq(repo, weights=None):
    """FROQ (https://github.com/LSIbabnikz/FROQ), AdaFace IR-101 of the repository and observer/observer_data.pkl, created
    with the repository's observer/initialization.py. Mean L2 norm of the selected intermediate layers."""
    import pickle
    import torch
    from PIL import Image
    _import(repo)
    from utils import parse_config_file, load_fr_model
    cwd = os.getcwd(); os.chdir(repo)
    try:
        fr, transform = load_fr_model(parse_config_file(str(Path(repo) / "face_recognition_model/adaface_config.yaml")))
    finally:
        os.chdir(cwd)
    fr = fr.cuda().eval()
    util, _, layer_set, corr = pickle.load(open(Path(repo) / "observer/observer_data.pkl", "rb"))
    positions = layer_set[:corr.index(max(corr)) + 1]
    layers = torch.argsort(torch.tensor(util), descending=True)[torch.tensor(positions)].tolist()
    mods, feats = list(fr.named_modules())[1:], {}
    for l in layers:
        mods[l][1].register_forward_hook(lambda m, i, o, l=l: feats.__setitem__(l, o.detach()))
    def fn(paths):
        fr(torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).cuda())
        n = len(paths)
        return (sum(torch.linalg.vector_norm(feats[l].reshape(n, -1), ord=2, dim=1).cpu() for l in layers) / len(layers)).numpy()
    return _batched(fn, 64)


def vitnt_fiqa(repo, weights):
    """ViTNT-FIQA (https://github.com/gurayozgur/ViTNT-FIQA), weights minchul_cvlface_adaface_vit_base_webface4m.pt."""
    import torch
    _import(repo)
    from backbones.vit.vit import VisionTransformer
    net = VisionTransformer(img_size=112, patch_size=8, num_classes=512, embed_dim=512, depth=24, mlp_ratio=3, num_heads=16,
                            drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.0)
    sd = torch.load(weights, map_location="cpu"); sd = sd.get("state_dict", sd.get("model", sd))
    net.load_state_dict({k[7:] if k.startswith("module.") else k[4:] if k.startswith("net.") else k: v for k, v in sd.items()},
                        strict=False)
    net = net.cuda().eval()
    return _batched(lambda p: net.calculate_ntfiq(_tensor(p), scaling=8.0, use_attention_weights=True,
                                                  last_block_attention_only=True, blocks_to_use=list(range(20))).reshape(-1).cpu().numpy(), 32)


def prefiqs(repo, weights, rho=0.4):
    """PreFIQs: L2 distance between the normalized embeddings of the ArcFace iResNet-100 (resnet100_ms1mv2_arcface.pth,
    iresnet code of https://github.com/jankolf/GraFIQs) and of its copy with rho = 40% global L1-unstructured pruning
    of all Conv2d and Linear weights. Native score: pruning drift (higher = more degradation, pi_m = -1)."""
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.prune as prune
    _import(repo)
    from backbones.iresnet import iresnet100
    orig = iresnet100(num_features=512, use_se=False); orig.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    orig.eval(); pruned = copy.deepcopy(orig)
    params = [(m, "weight") for m in pruned.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=rho)
    for m, n in params: prune.remove(m, n)
    orig, pruned = orig.cuda().eval(), pruned.cuda().eval()
    def fn(paths):
        x = _tensor(paths)
        quality = 2.0 - torch.linalg.vector_norm(F.normalize(orig(x), dim=1) - F.normalize(pruned(x), dim=1), dim=1)
        return 2.0 - quality.cpu().numpy().astype(np.float64)      # drift, from the float32 quality 2 - drift
    return _batched(fn, 128)


def attn_fiqa(repo, weights):
    """ATTN-FIQA (https://github.com/gurayozgur/ATTN-FIQA), weights vits_wf4m_adaface.pt (ViT-S, WebFace4M, AdaFace)."""
    import torch
    _import(repo)
    from backbones.vit.vit import VisionTransformer
    net = VisionTransformer(img_size=112, patch_size=8, num_classes=512, embed_dim=512, depth=12, mlp_ratio=5, num_heads=8,
                            drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.0)
    sd = torch.load(weights, map_location="cpu"); sd = sd.get("state_dict", sd.get("model", sd))
    net.load_state_dict({k[7:] if k.startswith("module.") else k[4:] if k.startswith("net.") else k: v for k, v in sd.items()},
                        strict=False)
    net = net.cuda().eval()
    return _batched(lambda p: net.calculate_attnfiqa(_tensor(p)).reshape(-1).float().cpu().numpy(), 64)


# ---------------------------------------------------------------- self-supervised (FR-integrated)
def magface(repo, weights):
    """MagFace (https://github.com/IrvingMeng/MagFace), weights magface_epoch_00025.pth. Native score: feature norm.
    Input in [0, 1] as in the official inference code."""
    import torch
    from types import SimpleNamespace
    _import(repo, "", "inference")                  # inference/ first, then the repository root
    from network_inf import builder_inf
    net = builder_inf(SimpleNamespace(arch="iresnet100", embedding_size=512, resume=str(weights), cpu_mode=False)).cuda().eval()
    return _batched(lambda p: torch.linalg.vector_norm(net(_tensor(p, "unit")), dim=1).cpu().numpy(), 128)


def pfe(repo, weights):
    """PFE (https://github.com/seasonSH/Probabilistic-Face-Embeddings), model folder PFE_sphere64_msarcface_am
    (graph.meta, ckpt-12000). RGB, official 112x96 center crop and (x - 127.5) / 128. Native score: sum of sigma^2."""
    import tensorflow as tf
    from tensorflow.core.protobuf import meta_graph_pb2
    tf.compat.v1.disable_eager_execution()
    mg = meta_graph_pb2.MetaGraphDef()
    mg.ParseFromString(tf.io.gfile.GFile(str(Path(weights) / "graph.meta"), "rb").read())
    for node in mg.graph_def.node:
        node.attr.pop("_output_shapes", None)
    g = tf.Graph()
    with g.as_default():
        saver = tf.compat.v1.train.import_meta_graph(mg, clear_devices=True)
    cfg = tf.compat.v1.ConfigProto(); cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=g, config=cfg); saver.restore(sess, str(Path(weights) / "ckpt-12000"))
    t = {k: g.get_tensor_by_name(k + ":0") for k in ("images", "phase_train", "keep_prob", "sigma_sq")}
    def crop(p):
        im = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); h0, w0 = im.shape[:2]
        y, x = int(round(0.5 * (h0 - 112))), int(round(0.5 * (w0 - 96)))
        return (im[y:y + 112, x:x + 96].astype(np.float32) - 127.5) / 128.0
    def run(paths):
        return np.concatenate([np.sum(sess.run(t["sigma_sq"], {t["images"]: np.asarray([crop(p) for p in paths[i:i + 128]]),
                                                               t["phase_train"]: False, t["keep_prob"]: 1.0}), axis=1)
                               for i in range(0, len(paths), 128)])
    return run


def cr_fiqa_l(repo, weights):
    """CR-FIQA(L) (https://github.com/fdbtrs/CR-FIQA), weights 181952backbone.pth."""
    import torch
    _import(repo)
    from backbones.iresnet import iresnet100
    net = iresnet100(num_features=512, qs=1, use_se=False); net.load_state_dict(torch.load(weights, map_location="cpu"))
    net = net.cuda().eval()
    return _batched(lambda p: net(_tensor(p))[1].reshape(-1).cpu().numpy(), 128)


def _vitfiqa_token(repo, weights, exit_block=None):
    import torch
    _import(repo)
    from backbones.vit_qs import VisionTransformer
    kw = {} if exit_block is None else {"early_exit_block": exit_block}
    net = VisionTransformer(img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=12, num_heads=8,
                            drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode="token", **kw)
    net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    for k in [k for k in sys.modules if k == "backbones" or k.startswith("backbones.")]:
        sys.modules.pop(k)
    sys.path.pop(0)
    return net.cuda().eval()


def vit_fiqa_t(repo, weights):
    """ViT-FIQA(T) (https://github.com/atzoriandrea/ViT-FIQA-Assessing-Face-Image-Quality-using-Vision-Transformers),
    quality-token checkpoint."""
    net = _vitfiqa_token(repo, weights)
    return _batched(lambda p: net(_tensor(p))[1].reshape(-1).cpu().numpy(), 64)


def carpm_fiqa_l(repo, weights):
    """CARPM-FIQA(L) (https://github.com/gurayozgur/CARPM-FIQA), weights CARPM-FIQA(L).pth."""
    import torch
    _import(repo)
    from backbones.iresnet import iresnet100
    net = iresnet100(num_features=512, qs=1); net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    net = net.cuda().eval()
    return _batched(lambda p: net(_tensor(p))[1].reshape(-1).float().cpu().numpy(), 64)


def ex_fiqa_fw_t(repo, weights):
    """EX-FIQA-FW(T) (https://github.com/gurayozgur/EX-FIQA) on the ViT-FIQA(T) checkpoint: fusion of the quality scores
    of all twelve early exits, s = sum_l (l / 78) s_l, l = 1..12."""
    nets = [_vitfiqa_token(repo, weights, exit_block=k) for k in range(12)]
    w = np.arange(1, 13) / 78.0
    return _batched(lambda p: np.stack([n(_tensor(p))[1].reshape(-1).float().cpu().numpy() for n in nets], 1) @ w, 64)


LOADERS = {f.__name__: f for f in (faceqnet, sdd_fiqa, clib_fiqa, diffiqa_r, ediffiqa_l, ser_fiq, grafiqs_l, froq,
                                    vitnt_fiqa, prefiqs, attn_fiqa, magface, pfe, cr_fiqa_l, vit_fiqa_t, carpm_fiqa_l,
                                    ex_fiqa_fw_t)}
