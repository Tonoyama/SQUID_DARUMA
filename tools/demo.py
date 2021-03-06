import argparse
from array import array
import os
import copy
import os.path as osp
import time
import cv2
import torch
from playsound import playsound
import threading
from loguru import logger
import numpy as np
from PIL import Image
from playsound import playsound
from blessed import Terminal

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

READY_IMG = os.path.abspath("tools/image/ready.png")
COUNTDOWN_IMG = os.path.abspath("tools/image/countdown.png")
START_IMG = os.path.abspath("tools/image/start.png")
RUN_IMG = os.path.abspath("tools/image/run.png")
GOAL_IMG = os.path.abspath("tools/image/goal.png")
DROPOUT_IMG = os.path.abspath("tools/image/dropout.png")
END_IMG = os.path.abspath("tools/image/end.png")
READY_IMG = cv2.imread(READY_IMG)
COUNTDOWN_IMG = cv2.imread(COUNTDOWN_IMG)
START_IMG = cv2.imread(START_IMG)
RUN_IMG = cv2.imread(RUN_IMG)
GOAL_IMG = cv2.imread(GOAL_IMG)
DROPOUT_IMG = cv2.imread(DROPOUT_IMG)
END_IMG = cv2.imread(END_IMG)

BGM = os.path.abspath("tools/sound/pink_soldiers.mp3")
LEVELUP_SOUND = os.path.abspath("tools/sound/levelup.mp3")
ONI_SOUND = os.path.abspath("tools/sound/oni_sound.mp3")
COUNTDOWN_SOUND = os.path.abspath("tools/sound/countdown.mp3")
DOOM_SOUND = os.path.abspath("tools/sound/doom.mp3")

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def create_gamma_img(gamma, img):
    gamma_cvt = np.zeros((256,1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
    return cv2.LUT(img, gamma_cvt)


def imageflow(cap, predictor, current_time, args, timelimit):
    WINDOW_NAME = 'SQUID GAME:Red light, Green light'
    #cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    avg, avg_a= None, None
    thresh = None
    killed_id = []
    timestamp = time.time()
    while time.time()-timestamp<timelimit:
        ret_val, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)

            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                counter = 0
                white_area = 0
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                for i, tlwh in enumerate(online_tlwhs):
                    obj_id = int(online_ids[i])
                    #ymin:ymax,xmin:xmax
                    x1, y1, w, h = tlwh
                    intbox = tuple(map(int, (x1, y1, x1+w, y1+h)))
                    intbox = [0 if i < 0 else i for i in intbox]

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if avg is None:
                        avg = gray.copy().astype("float")
                        continue

                    # ??????????????????????????????????????????????????????
                    cv2.accumulateWeighted(gray, avg, 0.93)
                    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

                    if counter == 0:
                        thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
                    counter += 1
                    # print(cv2.countNonZero(thresh))
                    #??????????????????
                    thresh1 = thresh[intbox[1]:intbox[3],intbox[0]:intbox[2]]
                    whole_area = thresh1.size
                    #?????????????????????
                    white_area = cv2.countNonZero(thresh1)
                    white_area = white_area/whole_area*100

                    SOUND = os.path.abspath("tools/sound/shotgun.mp3")

                    BAKUHATSU_IMG = os.path.abspath("tools/image/bakuhatsu.png")
                    BAKUHATSU_IMG = cv2.imread(BAKUHATSU_IMG)
                    # ?????????????????????
                    BAKUHATSU_IMG = cv2.resize(BAKUHATSU_IMG, dsize=(int(w/5), int(h/12)))
                    height, width = BAKUHATSU_IMG.shape[:2]

                    image = img_info['raw_img']
                    # ?????????????????????????????????
                    img_h, img_w = image.shape[:2]

                    x = int(x1+w/3)
                    y = int(y1+h/3)
                    y_height = int(y+height)
                    x_width = int(x+width)

                    # white_area???????????????????????????obj_id?????????????????????
                    if white_area > 2 and obj_id not in killed_id:
                        playsound(SOUND, block=False)
                        killed_id.append(obj_id)
                    
                    # ???????????????????????????????????????????????????????????????????????????????????????
                    for i in killed_id:
                        if obj_id == i:

                            alpha = 0.2 # ????????????????????????
                            beta = 0    # ???????????????

                            image_height = img_h - y
                            image_width = img_w - x
                            # ?????????????????????
                            dark_image = image[intbox[1]:intbox[3],intbox[0]:intbox[2]]
                            res_image = cv2.convertScaleAbs(dark_image, alpha=alpha, beta=beta)
                            image[intbox[1]:intbox[3],intbox[0]:intbox[2]] = res_image

                            if y_height >= img_h and image_height > 0:
                                y_height = img_h
                                BAKUHATSU_IMG = cv2.resize(BAKUHATSU_IMG, dsize=(int(w/7), int(image_height)))
                                #image[int(y):int(img_h), int(x):int(x+width)] = BAKUHATSU_IMG
                            elif x_width >= img_w and image_width > 0:
                                x_width = img_w
                                BAKUHATSU_IMG = cv2.resize(BAKUHATSU_IMG, dsize=(int(image_width), int(h/15)))
                                print("?????????")
                                #image[int(y):int(y+height), int(x):int(img_w)] = BAKUHATSU_IMG
                            elif image_height <= 0 or y <= 0:
                                print("?????????")
                            elif image_width <= 0 or x <= 0:
                                print("?????????")
                            else:
                                image[int(y):int(y+height), int(x):int(x+width)] = BAKUHATSU_IMG

                gray_a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if avg_a is None:
                    avg_a = gray_a.copy().astype("float")
                    continue

                # ??????????????????????????????????????????????????????
                cv2.accumulateWeighted(gray_a, avg_a, 0.93)
                frameDelta = cv2.absdiff(gray_a, cv2.convertScaleAbs(avg_a))
                thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

                cv2.drawContours(frame, contours, -1, color=(0, 0, 255), thickness=2)

                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, online_scores, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                cv2.imshow(WINDOW_NAME, online_im)
                print("BBOX??????")
            else:
                timer.toc()
                online_im = img_info['raw_img']

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            
        else:
            break
        frame_id += 1
    return killed_id



def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    
    WINDOW_NAME = 'SQUID GAME:Red light, Green light'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        print("??????????????????")
        cv2.imshow(WINDOW_NAME, READY_IMG)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        elif ch == ord("a") or ch == ord("A"):
            WAIT_TIME = 5
            for i in range(0, WAIT_TIME+1):
                time.sleep(1)
                countdown_img_copy = COUNTDOWN_IMG.copy()
                cv2.putText(countdown_img_copy, '%d' % (WAIT_TIME - i),(370, 430), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), thickness=20)
                cv2.imshow(WINDOW_NAME, countdown_img_copy)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
                playsound(COUNTDOWN_SOUND, block=False)
                print("??????????????????????????? %d" % (WAIT_TIME - i))

            time.sleep(1)
            cv2.imshow(WINDOW_NAME, START_IMG)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            playsound(DOOM_SOUND, block=False)
            time.sleep(3)

            killed_id = 0
            cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
            for i in range(2):
                print("?????????????????????(Press G Key)")
                playsound(ONI_SOUND, block=False)
                print("?????????????????????????????????????????????")
                time.sleep(4.5)
                timestamp = time.time()
                killed_id = imageflow(cap, predictor, current_time, args,10)

            time.sleep(2)
            print("%d?????????" % int(len(killed_id)))
            dropout_img_copy = DROPOUT_IMG.copy()
            cv2.putText(dropout_img_copy, '%d' % int(len(killed_id)),(200, 330), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), thickness=20)
            cv2.imshow(WINDOW_NAME, dropout_img_copy)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            time.sleep(3)
            
            print("??????!")
            cv2.imshow(WINDOW_NAME, END_IMG)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            time.sleep(3)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
