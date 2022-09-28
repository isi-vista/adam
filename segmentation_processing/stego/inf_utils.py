import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from .crf import dense_crf


def inference_res(img, model):
    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode="bilinear", align_corners=False)

        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        linear_pred = dense_crf(img, linear_probs[0]).argmax(0)
        cluster_pred = dense_crf(img, cluster_probs[0]).argmax(0)

    linear_obj_ids = np.unique(linear_pred)
    cluster_obj_ids = np.unique(cluster_pred)

    linear_masks = []
    for m in linear_obj_ids:
        linear_masks.append(np.equal(m, linear_pred))

    cluster_masks = []
    for m in cluster_obj_ids:
        cluster_masks.append(np.equal(m, cluster_pred))

    linear_masks = np.array(linear_masks)
    cluster_masks = np.array(cluster_masks)

    linear_masks_torch = torch.from_numpy(linear_masks)
    cluster_masks_torch = torch.from_numpy(cluster_masks)

    linear_boxes = masks_to_boxes(linear_masks_torch)
    cluster_boxes = masks_to_boxes(cluster_masks_torch)

    linear_boxes = linear_boxes.cpu().numpy()
    cluster_boxes = cluster_boxes.cpu().numpy()

    return linear_masks, cluster_masks, linear_boxes, cluster_boxes
