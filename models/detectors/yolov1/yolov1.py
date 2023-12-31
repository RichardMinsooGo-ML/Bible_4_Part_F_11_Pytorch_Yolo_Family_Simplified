import torch
import torch.nn as nn
import numpy as np

from utils.misc import multiclass_nms

from .yolov1_backbone import build_backbone
from .yolov1_neck import build_neck
from .yolov1_head import build_head


# YOLOv1
class YOLOv1(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 img_size=None,
                 num_classes=20,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 trainable=False,
                 deploy=False,
                 nms_class_agnostic :bool = False):
        super(YOLOv1, self).__init__()
        # ------------------------- Basic parameters  ---------------------------
        self.cfg                = cfg                  # Model configuration file
        self.img_size           = img_size             # Enter image size
        self.device             = device               # cuda or cpu
        self.num_classes        = num_classes          # number of classes
        self.trainable          = trainable            # training mark
        self.conf_thresh        = conf_thresh          # score threshold
        self.nms_thresh         = nms_thresh           # NMS threshold
        self.stride             = 32                   # The maximum stride size of the network
        self.deploy             = deploy
        self.nms_class_agnostic = nms_class_agnostic
        
        # ----------------------- Model network structure -----------------------
        ## backbone network
        self.backbone, feat_dim = build_backbone(
            cfg['backbone'], trainable&cfg['pretrained'])

        ## neck network
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim  = self.neck.out_dim

        ## Detection head
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        ## prediction layer
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)
    

    def create_grid(self, fmp_size):
        """ 
            Used to generate a G matrix, where each element is a pixel coordinate on the feature map.
        """
        # The width and height of the feature map
        ws, hs = fmp_size

        # Generate the x and y coordinates of the grid
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # Put together the coordinates of the two parts xy：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        return grid_xy


    def decode_boxes(self, pred, fmp_size):
        """
            Convert txtytwth to the commonly used x1y1x2y2 form。
        """
        # Generate grid coordinate matrix
        grid_cell = self.create_grid(fmp_size)

        # Calculate the center point coordinates, width and height of the predicted bounding box
        pred_ctr = (torch.sigmoid(pred[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred[..., 2:]) * self.stride

        # Convert the center coordinates, width and height of all bboxes into x1y1x2y2 form
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # Backbone network
        feat = self.backbone(x)

        # Neck network
        feat = self.neck(feat)

        # Detection head
        cls_feat, reg_feat = self.head(feat)

        # 预测层
        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        # Make some view adjustments to the size of pred to facilitate subsequent processing.
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # When testing, the author defaults to batch 1。
        # Therefore, we do not need to use the batch dimension and use [0] to remove it。
        obj_pred = obj_pred[0]       # [H*W, 1]
        cls_pred = cls_pred[0]       # [H*W, NC]
        reg_pred = reg_pred[0]       # [H*W, 4]

        # Score for each bounding box
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())
        
        # Solve the bounding box and normalize the bounding box: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)
        
        if self.deploy:
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            # Put predictions on cpu processing for post-processing
            scores = scores.cpu().numpy()
            bboxes = bboxes.cpu().numpy()
            
            # Post-processing
            bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels


    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            # Backbone network
            feat = self.backbone(x)

            # Neck network
            feat = self.neck(feat)

            # Detection head
            cls_feat, reg_feat = self.head(feat)

            # Prediction layer
            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # Make some view adjustments to the size of pred to facilitate subsequent processing.
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # Decode bbox
            box_pred = self.decode_boxes(reg_pred, fmp_size)

            # Network output
            outputs = {"pred_obj": obj_pred,                  # (Tensor) [B, M, 1]
                       "pred_cls": cls_pred,                   # (Tensor) [B, M, C]
                       "pred_box": box_pred,                   # (Tensor) [B, M, 4]
                       "stride": self.stride,                  # (Int)
                       "fmp_size": fmp_size                    # (List) [fmp_h, fmp_w]
                       }           
            return outputs
        
