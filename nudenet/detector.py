import os
import cv2
import pydload
import logging
import numpy as np
import onnxruntime
from progressbar import progressbar
from collections import defaultdict

from .detector_utils import preprocess_image, draw_labels_and_boxes
from .video_utils import get_interest_frames_from_video, get_original_params


def dummy(x):
    return x


FILE_URLS = {
    "default": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint.onnx",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_classes",
    },
    "base": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_checkpoint.onnx",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_classes",
    },
}


class Detector:
    detection_model = None
    classes = None
    video_extension = None

    def __init__(self, model_name="default"):
        """
        model = Detector()
        """
        checkpoint_url = FILE_URLS[model_name]["checkpoint"]
        classes_url = FILE_URLS[model_name]["classes"]

        home = os.path.expanduser("~")
        model_folder = os.path.join(home, f".NudeNet/")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        checkpoint_name = os.path.basename(checkpoint_url)
        checkpoint_path = os.path.join(model_folder, checkpoint_name)
        classes_path = os.path.join(model_folder, "classes")

        if not os.path.exists(checkpoint_path):
            print("Downloading the checkpoint to", checkpoint_path)
            pydload.dload(checkpoint_url, save_to_path=checkpoint_path, max_time=None)

        if not os.path.exists(classes_path):
            print("Downloading the classes list to", classes_path)
            pydload.dload(classes_url, save_to_path=classes_path, max_time=None)

        self.detection_model = onnxruntime.InferenceSession(checkpoint_path)

        self.classes = [c.strip() for c in open(classes_path).readlines() if c.strip()]

    @staticmethod
    def get_frames(video_path, filter_similar=True):
        """
        Wrapper for get all or only interested frames
        :param video_path: path ot video
        :param filter_similar: condition for turn onn / off similar frames filter
        :return: all or only interests frames with video meta data ( fps, length )
        """
        args = {} if filter_similar else {'skip_n_frames': 0, 'frame_similarity_threshold': 1}
        return get_interest_frames_from_video(video_path, **args)

    def detect_video(self, video_path, mode="default", min_prob=0.6, batch_size=2,
                     show_progress=True, filter_similar=True, output_path=None):
        frame_indices, frames, fps, video_length, width, height = self.get_frames(video_path, filter_similar)
        logging.debug(
            f"VIDEO_PATH: {video_path}, FPS: {fps}, Important frame indices: {frame_indices}, Video length: {video_length}, Video size: {width}x{height}"
        )

        if output_path is not None:
            fourcc_code = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print(fourcc_code, fps, (width, height))
            video_writer = cv2.VideoWriter(output_path, fourcc_code, fps, (width, height))

        if mode == "fast":
            frames = [
                preprocess_image(frame, min_side=480, max_side=800) for frame in frames
            ]
        else:
            frames = [preprocess_image(frame) for frame in frames]

        scale = frames[0][1]
        frames = [frame[0] for frame in frames]
        all_results = {
            "metadata": {
                "fps": fps,
                "video_length": video_length,
                "video_path": video_path,
            },
            "preds": defaultdict(list),
        }

        progress_func = progressbar

        if not show_progress:
            progress_func = dummy

        for _ in progress_func(range(int(len(frames) / batch_size) + 1)):
            batch = frames[:batch_size]
            batch_indices = frame_indices[:batch_size]
            frames = frames[batch_size:]
            frame_indices = frame_indices[batch_size:]
            if batch_indices:
                outputs = self.detection_model.run(
                    [s_i.name for s_i in self.detection_model.get_outputs()],
                    {self.detection_model.get_inputs()[0].name: np.asarray(batch)},
                )

                labels = [op for op in outputs if op.dtype == "int32"][0]
                scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
                boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

                # boxes /= scale
                for frame_index, frame_boxes, frame_scores, frame_labels, frame in zip(
                        frame_indices, boxes, scores, labels, frames
                ):

                    for box, score, label in zip(
                            frame_boxes, frame_scores, frame_labels
                    ):
                        if score < min_prob:
                            continue
                        scaling_box = box / scale
                        scaling_box = scaling_box.astype(int).tolist()
                        label = self.classes[label]

                        all_results["preds"][frame_index].append(
                            {
                                "box": [int(c) for c in scaling_box],
                                "score": float(score),
                                "label": label,
                            }
                        )
                        if output_path is not None:
                            frame_with_prediction_info = draw_labels_and_boxes(frame, box.astype(int).tolist(), label)
                            video_writer.write(frame_with_prediction_info)
        if output_path is not None:
            video_writer.release()
        return all_results

    def detect(self, img_path, mode="default", min_prob=None):
        if mode == "fast":
            image, scale = preprocess_image(img_path, min_side=480, max_side=800)
            if not min_prob:
                min_prob = 0.5
        else:
            image, scale = preprocess_image(img_path)
            if not min_prob:
                min_prob = 0.6

        outputs = self.detection_model.run(
            [s_i.name for s_i in self.detection_model.get_outputs()],
            {self.detection_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
        )

        labels = [op for op in outputs if op.dtype == "int32"][0]
        scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
        boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = self.classes[label]
            processed_boxes.append(
                {"box": [int(c) for c in box], "score": float(score), "label": label}
            )

        return processed_boxes

    def censor(self, img_path, out_path=None, visualize=False, parts_to_blur=[]):
        if not out_path and not visualize:
            print(
                "No out_path passed and visualize is set to false. There is no point in running this function then."
            )
            return

        image = cv2.imread(img_path)
        boxes = self.detect(img_path)

        if parts_to_blur:
            boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
        else:
            boxes = [i["box"] for i in boxes]

        for box in boxes:
            part = image[box[1]: box[3], box[0]: box[2]]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
            )

        if visualize:
            cv2.imshow("Blurred image", image)
            cv2.waitKey(0)

        if out_path:
            cv2.imwrite(out_path, image)


if __name__ == "__main__":
    m = Detector()
    print(m.detect("/Users/bedapudi/Desktop/n2.jpg"))
