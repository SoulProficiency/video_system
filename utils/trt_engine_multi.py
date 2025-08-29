import time

import tensorrt as trt
from cuda import cudart
import numpy as np
import cv2
import threading
from typing import List, Dict, Optional, Tuple
import logging
from . import common
# 设置日志
logger = logging.getLogger("TrtModel")


class TrtModel(object):
    # 类级别的锁，用于保护CUDA资源初始化
    _cuda_lock = threading.Lock()

    # 类级别的引擎缓存，避免重复加载相同模型
    _engine_cache = {}

    def __init__(self, engine_path, confidence_threshold: float = 0.5,
                 classes: Optional[List[int]] = None):
        # 初始化 allocations 列表，确保即使在出错时也能安全释放
        self.allocations = []

        try:
            # 使用锁保护CUDA资源初始化
            with TrtModel._cuda_lock:
                self.mean = None
                self.std = None
                self.confidence_threshold = confidence_threshold
                self.classes = classes
                self.model_name = engine_path.split("/")[-1].split(".")[0]
                self.n_classes = 80
                self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                                    'boat',
                                    'traffic light',
                                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                                    'horse',
                                    'sheep', 'cow',
                                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                    'suitcase', 'frisbee',
                                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                                    'skateboard',
                                    'surfboard',
                                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                    'banana',
                                    'apple',
                                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                                    'chair',
                                    'couch',
                                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                    'keyboard', 'cell phone',
                                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                                    'scissors',
                                    'teddy bear',
                                    'hair drier', 'toothbrush']

                self.input_shape = (640, 640)

                # 检查是否已缓存引擎
                if engine_path in TrtModel._engine_cache:
                    self.engine = TrtModel._engine_cache[engine_path]
                    logger.info(f"Using cached engine for {engine_path}")
                else:
                    # 使用不同的变量名来避免与标准日志记录器冲突
                    trt_logger = trt.Logger(trt.Logger.WARNING)
                    trt_logger.min_severity = trt.Logger.Severity.ERROR
                    runtime = trt.Runtime(trt_logger)
                    trt.init_libnvinfer_plugins(trt_logger, '')  # initialize TensorRT plugins
                    with open(engine_path, "rb") as f:
                        serialized_engine = f.read()
                    self.engine = runtime.deserialize_cuda_engine(serialized_engine)
                    TrtModel._engine_cache[engine_path] = self.engine
                    logger.info(f"Loaded and cached engine for {engine_path}")

                self.imgsz = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))[
                             2:]  # get the read shape of model, in case user input it wrong

                # 为每个实例创建独立的执行上下文
                self.context = self.engine.create_execution_context()

                # Setup I/O bindings - 每个实例有自己的绑定
                self.inputs = []
                self.outputs = []

                for i in range(self.engine.num_io_tensors):
                    name = self.engine.get_tensor_name(i)
                    dtype = self.engine.get_tensor_dtype(name)
                    shape = self.engine.get_tensor_shape(name)
                    is_input = False
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        is_input = True
                    if is_input:
                        self.batch_size = shape[0]
                    size = np.dtype(trt.nptype(dtype)).itemsize
                    for s in shape:
                        size *= s
                    allocation = common.cuda_call(cudart.cudaMalloc(size))
                    binding = {
                        'index': i,
                        'name': name,
                        'dtype': np.dtype(trt.nptype(dtype)),
                        'shape': list(shape),
                        'allocation': allocation,
                        'size': size
                    }
                    self.allocations.append(allocation)
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        self.inputs.append(binding)
                    else:
                        self.outputs.append(binding)

                # 实例级别的锁，保护实例特定的资源
                self._instance_lock = threading.Lock()

        except Exception as e:
            # 如果初始化失败，清理已分配的资源
            self._cleanup()
            raise e

    def _cleanup(self):
        """清理已分配的资源"""
        with TrtModel._cuda_lock:
            for allocation in self.allocations:
                try:
                    common.cuda_call(cudart.cudaFree(allocation))
                except Exception as e:
                    logger.warning(f"Failed to free CUDA memory: {e}")
            self.allocations = []

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, img):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # 使用实例级别的锁保护推理过程
        with self._instance_lock:
            # Prepare the output data.
            outputs = []
            for shape, dtype in self.output_spec():
                outputs.append(np.zeros(shape, dtype))

            # Process I/O and execute the network.
            common.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(img))

            self.context.execute_v2(self.allocations)
            for o in range(len(outputs)):
                common.memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])
            return outputs

    def preprocess(self, image):
        """预处理图像"""
        # 获取原始图像尺寸
        h, w = image.shape[:2]

        # 计算缩放比例
        scale = min(self.input_shape[0] / h, self.input_shape[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # 调整图像大小
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建填充后的图像
        padded = np.full((self.input_shape[0], self.input_shape[1], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # 转换颜色空间和数据类型
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.transpose(2, 0, 1)  # HWC to CHW
        padded = np.ascontiguousarray(padded, dtype=np.float32) / 255.0

        return padded, scale, (w, h)

    def draw_detections(self, image, detections, conf_threshold=0.5):
        """在图像上绘制检测结果"""
        for det in detections:
            if det['confidence'] < conf_threshold:
                continue

            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            label = f"{det['class_name']}: {det['confidence']:.2f}"

            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 绘制文本背景
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline),
                (x1 + text_width, y1),
                color,
                -1
            )

            # 绘制文本
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return image

    def predict(self, frame, conf=0.5, show=True):
        t1 = time.time()
        input_data, scale, orig_shape = self.preprocess(frame)

        data = self.infer(input_data)
        detections = self.postprocess(output=data[0], scale=scale, orig_shape=orig_shape, conf_threshold=conf)
        return detections

    def postprocess(self, output, scale, orig_shape, conf_threshold=0.5, nms_threshold=0.4):
        """后处理输出"""
        # 重塑输出为 (8400, 84)
        output = output[0].transpose(1, 0)

        # 提取边界框和类别分数
        boxes = output[:, :4]  # x_center, y_center, width, height
        scores = output[:, 4:]  # 类别分数
        # 找到每个框的最高分数和对应的类别
        class_ids = np.argmax(scores, axis=1)

        confidences = np.max(scores, axis=1)

        # 应用置信度阈值
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # 将中心坐标转换为角坐标
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x_center - width / 2) / scale
        y1 = (y_center - height / 2) / scale
        x2 = (x_center + width / 2) / scale
        y2 = (y_center + height / 2) / scale

        boxes = np.column_stack([x1, y1, x2, y2])

        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            conf_threshold,
            nms_threshold
        )

        # 准备最终检测结果
        detections = []
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            for idx in indices:
                x1, y1, x2, y2 = boxes[idx]
                # 确保边界框在图像范围内
                x1 = max(0, min(x1, orig_shape[0]))
                y1 = max(0, min(y1, orig_shape[1]))
                x2 = max(0, min(x2, orig_shape[0]))
                y2 = max(0, min(y2, orig_shape[1]))

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidences[idx],
                    'class_id': class_ids[idx],
                    'class_name': self.class_names[class_ids[idx]]
                })

        return detections

    def get_fps(self):
        import time
        img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100 / (time.perf_counter() - t0), 'FPS')

    def __del__(self):
        """析构函数，释放CUDA资源"""
        # 确保 allocations 属性存在
        if hasattr(self, 'allocations'):
            self._cleanup()