import numpy as np
import torch

INSTANCE_CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COLORS = np.random.uniform(0, 255, size=(len(INSTANCE_CATEGORIES), 3))


def inference_res(image, model, threshold):
    with torch.no_grad():
        outputs = model(image)

    # label likelihood scores
    scores = list(outputs[0]["scores"].detach().cpu().numpy())
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)

    # threshold filter masks
    masks = (outputs[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[:thresholded_preds_count]

    # retrieve bounding boxes in form (x1, y1), (x2, y2)
    boxes = [
        [(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]["boxes"].detach().cpu()
    ]

    # filter results using likelihood threshold
    boxes = boxes[:thresholded_preds_count]

    # class labels found from all mask-rcnn labels identified
    labels = [INSTANCE_CATEGORIES[i] for i in outputs[0]["labels"]]

    return masks, boxes, labels
