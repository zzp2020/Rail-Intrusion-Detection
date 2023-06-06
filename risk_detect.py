# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python risk_detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python risk_detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

#导入轨道识别所需要的包
import time
import cv2
import numpy as np
from scipy.special import comb


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5m.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam'E:/Peiru_Chen/Railway_detection/imageP'
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpupython detect.py --weights yolov5m.pt --source "E:/Download/RC1.png"
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    first = True
    a1 = 1
    last_inter_x, last_left_x, last_right_x,last_object = [],[],[],[]

    def bezier_curve(points, ntimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.
           points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            ntimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        def bernstein_poly(i, n, t):
            """
             The Bernstein polynomial of n, i as a function of t
            """
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

        points = points[:-5]
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, ntimes)

        polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals.astype('int32'), yvals.astype('int32')

    def fit(leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit

    def draw_area1(image, left_x, right_x, left_y, right_y):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        newwarp = cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

    def draw_area2(image, left_x, right_x, left_y, right_y):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        newwarp = cv2.fillPoly(warp_zero, np.int_([pts]), (0, 0, 255))

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result
    def objectcolor(x):
        if x == 4 :
            color = (0, 0, 225)
            return color
        elif x == 3.5:
            color = (0, 128, 225)
            return color
        elif x == 3:
            color = (0, 255, 225)
            return color
        elif x == 2.5:
            color = (255, 0, 0)
            return color
        elif x == 2:
            color = (0, 255, 0)
            return color
    sev_o = {'0': 2, '1': 2, '2': 1}
    sev_a = {'danger': 2, 'closing': 1.5, 'risk': 1 }
    level = {'2':'I', '2.5':'II', '3':'III', '3.5':'IV', '4':'V'}

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        frame = im0s.copy()
        # initialization for line detection
        expt_startLeft = 450
        expt_startRight = 840
        expt_startTop = 330
        # value initialize
        left_maxpoint = [0] * 50
        right_maxpoint = [195] * 50

        # convolution filter
        kernel = np.array([
            [-1, 1, 0, 1, -1],
            [-1, 1, 0, 1, -1],
            [-1, 1, 0, 1, -1],
            [-1, 1, 0, 1, -1],
            [-1, 1, 0, 1, -1]
        ])
        start = time.time()
        # cut away invalid frame area
        valid_frame = frame[expt_startTop:, expt_startLeft:expt_startRight]
        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)
        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)
        # apply gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)
        # merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # initialization for sliding window property
        sliding_window = [20, 190, 200, 370]
        slide_interval = 15
        slide_height = 15
        slide_width = 60

        # initialization for bezier curve variables
        left_points = []
        right_points = []

        # define count value
        count = 0
        for i in range(340, 40, -slide_interval):
            # get edges in sliding window
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

            # left railroad line processing
            if left_edge.argmax() > 0:
                left_maxindex = sliding_window[0] + left_edge.argmax()
                left_maxpoint[count] = left_maxindex
                #cv2.line(valid_frame, (left_maxindex, i + int(slide_height / 2)),(left_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                left_points.append([left_maxindex, i + int(slide_height / 2)])
                sliding_window[0] = max(0, left_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[1] = min(390, left_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                #cv2.rectangle(valid_frame, (sliding_window[0], i + slide_height), (sliding_window[1], i), (0, 255, 0), 1)

            # right railroad line processing
            if right_edge.argmax() > 0:
                right_maxindex = sliding_window[2] + right_edge.argmax()
                right_maxpoint[count] = right_maxindex
                #cv2.line(valid_frame, (right_maxindex, i + int(slide_height / 2)), (right_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                right_points.append([right_maxindex, i + int(slide_height / 2)])
                sliding_window[2] = max(0, right_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[3] = min(390, right_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                #cv2.rectangle(valid_frame, (sliding_window[2], i + slide_height), (sliding_window[3], i), (0, 0, 255), 1)
            count += 1
        intersection_point = None
        # bezier curve process
        bezier_left_xval, bezier_left_yval = bezier_curve(left_points, 50)
        bezier_right_xval, bezier_right_yval = bezier_curve(right_points, 50)

        bezier_left_xval,bezier_right_xval=bezier_left_xval+450,bezier_right_xval+450
        bezier_left_yval, bezier_right_yval = bezier_left_yval + 330, bezier_right_yval + 330                           #第一危险区
        rail_halflength = (bezier_right_xval - bezier_left_xval)/1067*2200/2
        risk_left_xval, risk_right_xval =bezier_left_xval - rail_halflength , bezier_right_xval + rail_halflength       #第二危险区
        left_fit,right_fit = fit(bezier_left_xval, bezier_left_yval,bezier_right_xval, bezier_right_yval)
        risk_left_fit,risk_right_fit = fit(risk_left_xval, bezier_left_yval,risk_right_xval, bezier_right_yval)
        ploty = np.arange(720,360,-5)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        left_fitx = left_fitx.astype(int)
        right_fitx = right_fitx.astype(int)
        risk_left_fitx = risk_left_fit[0] * ploty ** 2 + risk_left_fit[1] * ploty + risk_left_fit[2]
        risk_right_fitx = risk_right_fit[0] * ploty ** 2 + risk_right_fit[1] * ploty + risk_right_fit[2]
        inter_point = right_fitx - left_fitx
        if np.min(inter_point) <= 0:
            index_point = np.sum(inter_point >= 5)
            intersection_point = int(ploty[index_point-1])
            ploty = ploty[:index_point]
        end = time.time()
        left_fitx = left_fitx[:len(ploty)]
        right_fitx = right_fitx[:len(ploty)]
        risk_left_fitx = risk_left_fitx[:len(ploty)]
        risk_right_fitx = risk_right_fitx[:len(ploty)]
        try:
            media = draw_area1(frame, left_fitx, right_fitx, ploty, ploty)
            result = draw_area2(media, risk_left_fitx, risk_right_fitx, ploty, ploty)
        except IndexError:
            pass



        #异物识别
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, result.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            interval = 0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    obj_left_x, obj_right_x, inter_x, obj_Y = int(xyxy[0]), int(xyxy[2]), int(
                                xyxy[0] + (xyxy[2] - xyxy[0]) / 2), int(xyxy[3])
                    danger_left_x = left_fit[0] * obj_Y ** 2 + left_fit[1] * obj_Y + left_fit[2]
                    danger_right_x = right_fit[0] * obj_Y ** 2 + right_fit[1] * obj_Y + right_fit[2]
                    risk_left_x = risk_left_fit[0] * obj_Y ** 2 + risk_left_fit[1] * obj_Y + risk_left_fit[2]
                    risk_right_x = risk_right_fit[0] * obj_Y ** 2 + risk_right_fit[1] * obj_Y + risk_right_fit[
                                2]

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        if obj_Y >= 360 and (intersection_point == None or obj_Y >= intersection_point):
                            strat_1 = time.time()
                            if (risk_left_x < obj_left_x and obj_left_x < risk_right_x) or (risk_left_x < obj_right_x and obj_right_x < risk_right_x) or (risk_left_x < inter_x and inter_x < risk_right_x):
                                if (danger_left_x < obj_left_x and obj_left_x < danger_right_x) or (
                                        danger_left_x < obj_right_x and obj_right_x < danger_right_x) or (
                                        danger_left_x < inter_x and inter_x < danger_right_x):
                                    label = None if hide_labels else (
                                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    Grade = "Risk grade -- " + level[str(sev_a["danger"]+sev_o[str(c)])]
                                    annotator.box_label(xyxy, label, color=objectcolor(sev_a["danger"]+sev_o[str(c)]))
                                    annotator.drawtext(color=objectcolor(sev_a["danger"]+sev_o[str(c)]), text=Grade)
                                else:
                                    if a1 / 30 != 0:
                                        a1 += 1
                                        last_inter_x.append(inter_x)
                                        last_left_x.append(danger_left_x)
                                        last_right_x.append(danger_right_x)
                                        last_object.append(c)
                                        label = None if hide_labels else (
                                                    names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                        Grade = "Risk grade -- " + level[str(sev_a["risk"]+sev_o[str(c)])]
                                        annotator.box_label(xyxy, label, color=objectcolor(sev_a["risk"]+sev_o[str(c)]))
                                        annotator.drawtext(color=objectcolor(sev_a["risk"] + sev_o[str(c)]),text=Grade)

                                    else:
                                        if (abs(last_right_x[a1 - 30] - last_inter_x[a1 - 30])) / (
                                                last_right_x[a1 - 30] - last_left_x[a1 - 30]) - (abs(danger_right_x - inter_x)) / (
                                                danger_right_x - danger_left_x) <= -0.5 and (
                                                c == last_object[a1 - 30]):

                                                label = None if hide_labels else (
                                                        names[
                                                            c] if hide_conf else f'{names[c]} {conf:.2f}')
                                                Grade = "Risk grade -- " + level[str(sev_a["closing"]+sev_o[str(c)])]
                                                annotator.box_label(xyxy, label, color=objectcolor(sev_a["closing"]+sev_o[str(c)]))
                                                annotator.drawtext(color=objectcolor(sev_a["closing"]+sev_o[str(c)]), text=Grade)
                                        else:
                                                label = None if hide_labels else (
                                                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                                Grade = "Risk grade -- " + level[str(sev_a["risk"]+sev_o[str(c)])]
                                                annotator.box_label(xyxy, label, color=objectcolor(sev_a["risk"]+sev_o[str(c)]))
                                                annotator.drawtext(color=objectcolor(sev_a["risk"]+sev_o[str(c)]), text=str(Grade))

                                        last_inter_x.clear()
                                        last_left_x.clear()
                                        last_right_x.clear()
                                        last_object.clear()
                            end_1 = time.time()
                            interval = (end_1 - strat_1) * 1000
                        else:
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=(255,255,255))
                        LOGGER.info(f"{s}{''}{(dt[1].dt * 1E3 + (end - start) * 1000 + interval):.1f}ms")
                        with open("data/time_record.txt", 'a') as f:
                            f.write(f"{s}{''}{(dt[1].dt * 1E3 + (end - start) * 1000 + interval):.1f}ms" + '\n')

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                             BGR=True)

                        # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p),
                                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)


    '''
        # Print time (inference-only)
        if len(det):
            LOGGER.info(f"{s}{''}{(dt[1].dt * 1E3+(end-start)*1000+interval):.1f}ms")

    # Print results

    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    '''
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best_parameters/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '../dataset/experiment/*/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/dataset.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
