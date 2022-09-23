import cv2
import numpy as np
import argparse
import onnxruntime as ort

class centernet():
    def __init__(self, model_path, prob_threshold=0.4, iou_threshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        self.num_classes = len(self.classes)
        self.confThreshold = prob_threshold
        self.nmsThreshold = iou_threshold

        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_path, so)
        self.input_height = self.net.get_inputs()[0].shape[2]
        self.input_width = self.net.get_inputs()[0].shape[3]
        self.hm_h = self.net.get_outputs()[0].shape[2]
        self.hm_w = self.net.get_outputs()[0].shape[3]
        self.grid = self._make_grid(self.hm_w, self.hm_h)

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), axis=2).astype(np.float32)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        print(label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), 0, 0.7, (0, 255, 0), thickness=2)
        return frame

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detect(self, srcimg):
        img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height),
                         interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - self.mean) / (self.std)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        stride = self.input_height / outs[0].shape[2]
        score = np.max(outs[0].squeeze(axis=0), axis=0).reshape(-1)
        score = self.sigmoid(score)
        classid = np.argmax(outs[0].squeeze(axis=0), axis=0).reshape(-1)
        flag = score > self.confThreshold
        score = score[flag]
        classid = classid[flag]

        reg_xy_list = outs[1].squeeze(axis=0).transpose((1, 2, 0))
        reg_wh_list = outs[2].squeeze(axis=0).transpose((1, 2, 0))
        cx_cy = reg_xy_list + self.grid
        cx_cy = cx_cy.reshape(-1, 2)[flag]
        wh = reg_wh_list.reshape(-1, 2)[flag]

        boxes, confidences, classIds = [], [], []
        scale_h = srcimg.shape[0] / self.input_height
        scale_w = srcimg.shape[1] / self.input_width
        for i in range(cx_cy.shape[0]):
            x = max(0, int((cx_cy[i, 0] - 0.5 * wh[i, 0]) * stride * scale_w))
            y = max(0, int((cx_cy[i, 1] - 0.5 * wh[i, 1]) * stride * scale_h))
            width = min(srcimg.shape[1] - 1, int(wh[i, 0] * stride * scale_w))
            height = min(srcimg.shape[0] - 1, int(wh[i, 1] * stride * scale_h))

            boxes.append([x, y, width, height])
            classIds.append(classid[i])
            confidences.append(score[i])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            srcimg = self.drawPred(srcimg, classIds[i], confidences[i], left, top, left + width, top + height)
        return srcimg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/person.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='ctdet_coco_dlav0_384.onnx',
                        choices=["ctdet_coco_dlav0_384.onnx", "ctdet_coco_dlav0_512.onnx"],
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = centernet(args.modelpath, prob_threshold=args.confThreshold, iou_threshold=args.nmsThreshold)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
