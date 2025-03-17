# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch

class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolo11n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
        self.combine_mask = True if self.args.combine_mask else False

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        if isinstance(preds[1], tuple) and len(preds[1]) == 2:
            pred, proto, mask_pred = preds[0], preds[1][1], preds[2]
        else:
            pred, proto, mask_pred = preds[0], preds[1], preds[2]

        if not self.combine_mask:
            pred = torch.cat((pred, mask_pred), dim=1)

        return super().postprocess(pred, img, orig_imgs, mask_preds=mask_pred if self.combine_mask else None, protos=proto)

    def construct_results(self, preds, img, orig_imgs, protos, mask_preds=None):
        """
        Constructs a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            protos (List[torch.Tensor]): List of prototype masks.
            mask_pred(List[torch.Tensor]): List of predicted masks (optional).
                not None if self.combine_mask is True.

        Returns:
            (list): List of result objects containing the original images, image paths, class names, bounding boxes, and masks.
        """
        assert (mask_preds is not None) == self.combine_mask, "mask_preds should be None if self.combine_mask is False, or vise versa."
        if self.combine_mask:
            return [
                self.construct_result(pred, img, orig_img, img_path, proto, mask_pred)
                for pred, orig_img, img_path, proto, mask_pred in zip(preds, orig_imgs, self.batch[0], protos, mask_preds)
            ]
        else:
            return [
                self.construct_result(pred, img, orig_img, img_path, proto)
                for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
            ]

    def construct_result(self, pred, img, orig_img, img_path, proto, mask_pred=None):
        """
        Constructs the result object from the prediction.

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
                p, mask_pred if combine_mask else p
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.
            mask_pred (torch.Tensor): The predicted masks (optional).
                not None if self.combine_mask is True.
        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if not self.combine_mask:
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        else:
            masks = ops.process_combine_mask(mask_pred, pred[:, 5], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
