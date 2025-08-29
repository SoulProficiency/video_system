import cv2
import time
import logging
import threading
import numpy as np
from typing import Optional, Any, Dict, List, Tuple, Callable
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import queue
import asyncio
from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import Queue, Process, Value, Array
import copy

# 导入自定义模块
from utils.video_stream_decoder import VideoStreamDecoder
from utils.trt_engine_multi import TrtModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamDecoder")

# 全局变量
app = FastAPI(title="多路视频流管理系统")
stream_manager = None
display_thread = None
stopped = False
frame_queue = queue.Queue(maxsize=10)
active_streams = set()


# 数据模型
class StreamConfig(BaseModel):
    stream_url: str
    buffer_size: int = 102400
    hw_accel: Optional[str] = None
    timeout: int = 5000000
    reconnect_delay: int = 5
    max_retries: int = -1
    keyframe_only: bool = False
    frame_skip: int = 3  # 抽帧分析参数，每N帧分析一帧


class StreamUpdate(BaseModel):
    buffer_size: Optional[int] = None
    hw_accel: Optional[str] = None
    timeout: Optional[int] = None
    reconnect_delay: Optional[int] = None
    max_retries: Optional[int] = None
    keyframe_only: Optional[bool] = None
    frame_skip: Optional[int] = None  # 抽帧分析参数


class DetectionConfig(BaseModel):
    model_path: str
    confidence_threshold: float = 0.5
    classes: Optional[List[int]] = None


class AlertCondition(BaseModel):
    class_name: str
    min_confidence: float = 0.5
    min_count: int = 1
    max_count: Optional[int] = None
    cooldown: int = 5  # 报警冷却时间(秒)


class AlertAction(BaseModel):
    action_type: str  # "log", "http", "email", "mqtt"
    config: Dict[str, Any]


# 报警处理器
class AlertHandler:
    def __init__(self):
        self.alert_conditions = {}  # model_name -> List[AlertCondition]
        self.alert_actions = {}  # model_name -> List[AlertAction]
        self.last_alert_time = defaultdict(float)  # (stream_id, model_name, class_name) -> last_alert_time

    def add_alert_condition(self, model_name: str, condition: AlertCondition, action: AlertAction):
        """为模型添加报警条件和动作"""
        if model_name not in self.alert_conditions:
            self.alert_conditions[model_name] = []
            self.alert_actions[model_name] = []

        self.alert_conditions[model_name].append(condition)
        self.alert_actions[model_name].append(action)
        logger.info(f"Added alert condition for {model_name}: {condition.class_name}")

    def check_alerts(self, stream_id: str, model_name: str, detections: List[Dict]) -> List[Dict]:
        """检查检测结果是否触发报警条件"""
        alerts_triggered = []

        if model_name not in self.alert_conditions:
            return alerts_triggered

        # 按类别分组检测结果
        class_detections = defaultdict(list)
        for detection in detections:
            class_detections[detection["class_name"]].append(detection)

        # 检查每个报警条件
        for condition, action in zip(self.alert_conditions[model_name], self.alert_actions[model_name]):
            class_name = condition.class_name
            if class_name in class_detections:
                detections_for_class = class_detections[class_name]
                confidences = [d["confidence"] for d in detections_for_class]
                high_conf_detections = [d for d in detections_for_class if d["confidence"] >= condition.min_confidence]

                count = len(high_conf_detections)
                if (count >= condition.min_count and
                        (condition.max_count is None or count <= condition.max_count)):

                    # 检查冷却时间
                    alert_key = (stream_id, model_name, class_name)
                    current_time = time.time()
                    if current_time - self.last_alert_time[alert_key] >= condition.cooldown:
                        self.last_alert_time[alert_key] = current_time

                        alert_info = {
                            "stream_id": stream_id,
                            "model_name": model_name,
                            "class_name": class_name,
                            "count": count,
                            "detections": high_conf_detections,
                            "action": action,
                            "timestamp": datetime.datetime.now().isoformat()
                        }

                        # 执行报警动作
                        self.execute_alert_action(alert_info)
                        alerts_triggered.append(alert_info)

        return alerts_triggered

    def execute_alert_action(self, alert_info: Dict):
        """执行报警动作"""
        action = alert_info["action"]

        if action.action_type == "log":
            logger.warning(
                f"ALERT: {alert_info['class_name']} detected in {alert_info['stream_id']} "
                f"by {alert_info['model_name']} (count: {alert_info['count']})"
            )

        elif action.action_type == "http":
            # 这里可以实现HTTP请求逻辑
            logger.info(f"HTTP alert triggered: {alert_info}")

        elif action.action_type == "email":
            # 这里可以实现邮件发送逻辑
            logger.info(f"Email alert triggered: {alert_info}")

        elif action.action_type == "mqtt":
            # 这里可以实现MQTT发布逻辑
            logger.info(f"MQTT alert triggered: {alert_info}")


# 模型管理器
class ModelManager:
    def __init__(self):
        self.models = {}  # model_name -> TrtModel
        self.active_models = defaultdict(list)  # stream_id -> List[model_name]
        self.alert_handler = AlertHandler()
        # 使用线程池进行异步推理
        self.executor = ThreadPoolExecutor(max_workers=4)
        # 使用进程池进行CPU密集型推理任务
        self.process_executor = ProcessPoolExecutor(max_workers=2)

    def load_model(self, model_config: DetectionConfig) -> str:
        """加载TensorRT模型"""
        print("model path : ", model_config.model_path)
        model = TrtModel(
            model_config.model_path,
            model_config.confidence_threshold,
            model_config.classes
        )

        self.models[model.model_name] = model
        return model.model_name

    def add_model_to_stream(self, stream_id: str, model_name: str):
        """为视频流添加模型"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        if model_name not in self.active_models[stream_id]:
            self.active_models[stream_id].append(model_name)
            logger.info(f"Added model {model_name} to stream {stream_id}")

    def remove_model_from_stream(self, stream_id: str, model_name: str):
        """从视频流移除模型"""
        if model_name in self.active_models[stream_id]:
            self.active_models[stream_id].remove(model_name)
            logger.info(f"Removed model {model_name} from stream {stream_id}")

    def process_frame_async(self, stream_id: str, frame: np.ndarray, callback: Callable):
        """异步处理帧数据，应用所有激活的模型"""
        if stream_id not in self.active_models or not self.active_models[stream_id]:
            callback(frame.copy(), [])
            return

        # 提交到线程池进行异步处理
        future = self.executor.submit(self._process_frame_sync, stream_id, frame)
        future.add_done_callback(lambda f: callback(*f.result()))

    def _process_frame_sync(self, stream_id: str, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """同步处理帧数据"""
        result_frame = frame.copy()
        all_detections = []

        for model_name in self.active_models[stream_id]:
            model = self.models[model_name]
            detections = model.predict(frame)
            result_frame = model.draw_detections(result_frame, detections)

            # 检查报警条件
            self.alert_handler.check_alerts(stream_id, model_name, detections)

            all_detections.extend(detections)

        return result_frame, all_detections

    def add_alert_condition(self, model_name: str, condition: AlertCondition, action: AlertAction):
        """为模型添加报警条件"""
        self.alert_handler.add_alert_condition(model_name, condition, action)

    def get_stream_models(self, stream_id: str) -> List[str]:
        """获取视频流激活的模型列表"""
        return self.active_models.get(stream_id, [])

    def get_loaded_models(self) -> List[str]:
        """获取所有已加载的模型"""
        return list(self.models.keys())

    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class StreamManager:
    def __init__(self):
        self.decoders: Dict[str, VideoStreamDecoder] = {}
        self.model_manager = ModelManager()
        self.lock = threading.RLock()

    def add_stream(self, stream_id: str, stream_url: str, **kwargs) -> bool:
        """添加一路视频流"""
        with self.lock:
            if stream_id in self.decoders:
                logger.warning(f"Stream ID {stream_id} already exists.")
                return False

            decoder = VideoStreamDecoder(
                stream_id,
                stream_url,
                inference_callback=self.model_manager.process_frame_async,
                **kwargs
            )
            self.decoders[stream_id] = decoder
            logger.info(f"Added stream {stream_id} with URL {stream_url}")
            return True

    def remove_stream(self, stream_id: str) -> bool:
        """移除一路视频流"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return False

            decoder = self.decoders[stream_id]
            decoder.stop()
            time.sleep(0.1)
            del self.decoders[stream_id]
            logger.info(f"Removed stream {stream_id}")
            return True

    def start_stream(self, stream_id: str) -> bool:
        """启动指定视频流"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return False

            try:
                self.decoders[stream_id].start()
                return True
            except Exception as e:
                logger.error(f"Failed to start stream {stream_id}: {e}")
                return False

    def stop_stream(self, stream_id: str) -> bool:
        """停止指定视频流"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return False

            try:
                self.decoders[stream_id].stop()
                return True
            except Exception as e:
                logger.error(f"Failed to stop stream {stream_id}: {e}")
                return False

    def start_all(self):
        """启动所有视频流"""
        with self.lock:
            for stream_id, decoder in self.decoders.items():
                try:
                    decoder.start()
                except Exception as e:
                    logger.error(f"Failed to start stream {stream_id}: {e}")

    def stop_all(self):
        """停止所有视频流"""
        with self.lock:
            for stream_id, decoder in self.decoders.items():
                try:
                    decoder.stop()
                except Exception as e:
                    logger.error(f"Failed to stop stream {stream_id}: {e}")
            # 关闭模型管理器的执行器
            self.model_manager.shutdown()

    def get_frame(self, stream_id: str, processed: bool = False) -> Optional[np.ndarray]:
        """获取指定流的当前帧"""
        with self.lock:
            if stream_id not in self.decoders:
                return None
            return self.decoders[stream_id].get_frame(processed)

    def get_all_frames(self, processed: bool = False) -> Dict[str, Optional[np.ndarray]]:
        """获取所有流的当前帧"""
        frames = {}
        with self.lock:
            for stream_id, decoder in self.decoders.items():
                frames[stream_id] = decoder.get_frame(processed)
        return frames

    def update_stream_settings(self, stream_id: str, **kwargs):
        """更新指定流的设置"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return

            self.decoders[stream_id].update_settings(**kwargs)

    def get_stream_stats(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取指定流的统计信息"""
        with self.lock:
            if stream_id not in self.decoders:
                return None
            return self.decoders[stream_id].get_stats()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有流的统计信息"""
        stats = {}
        with self.lock:
            for stream_id, decoder in self.decoders.items():
                stats[stream_id] = decoder.get_stats()
        return stats

    def get_active_streams(self) -> List[str]:
        """获取当前活跃的流ID列表"""
        with self.lock:
            return list(self.decoders.keys())

    def enable_detection(self, stream_id: str, enable: bool):
        """启用或禁用指定流的检测功能"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return False

            self.decoders[stream_id].enable_detection(enable)
            return True

    def load_model(self, model_config: DetectionConfig) -> str:
        """加载模型"""
        return self.model_manager.load_model(model_config)

    def add_model_to_stream(self, stream_id: str, model_name: str):
        """为视频流添加模型"""
        self.model_manager.add_model_to_stream(stream_id, model_name)

    def remove_model_from_stream(self, stream_id: str, model_name: str):
        """从视频流移除模型"""
        self.model_manager.remove_model_from_stream(stream_id, model_name)

    def add_alert_condition(self, model_name: str, condition: AlertCondition, action: AlertAction):
        """为模型添加报警条件"""
        self.model_manager.add_alert_condition(model_name, condition, action)

    def get_stream_models(self, stream_id: str) -> List[str]:
        """获取视频流激活的模型列表"""
        return self.model_manager.get_stream_models(stream_id)

    def get_loaded_models(self) -> List[str]:
        """获取所有已加载的模型"""
        return self.model_manager.get_loaded_models()

    def switch_to_original_stream(self, stream_id: str):
        """切换到原始流（不进行推理）"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return False

            self.decoders[stream_id].enable_detection(False)
            logger.info(f"Switched to original stream for {stream_id}")
            return True

    def switch_to_processed_stream(self, stream_id: str):
        """切换到处理后的流（进行推理）"""
        with self.lock:
            if stream_id not in self.decoders:
                logger.warning(f"Stream ID {stream_id} does not exist.")
                return False

            self.decoders[stream_id].enable_detection(True)
            logger.info(f"Switched to processed stream for {stream_id}")
            return True


def display_loop():
    """显示循环，在主线程中运行"""
    global stopped, stream_manager, frame_queue, active_streams

    # 定义小屏幕尺寸和总分辨率
    SMALL_SCREEN_WIDTH = 856
    SMALL_SCREEN_HEIGHT = 480
    TOTAL_WIDTH = SMALL_SCREEN_WIDTH * 4  # 4列
    TOTAL_HEIGHT = SMALL_SCREEN_HEIGHT * 2  # 2行

    # 创建显示窗口
    cv2.namedWindow("Multi-Stream Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Stream Display", TOTAL_WIDTH, TOTAL_HEIGHT)

    # 定义布局 - 2x4网格 (2行，每行4个)
    layout = [
        ["cam1", "cam2", "cam3", "cam4"],
        ["cam5", "cam6", "cam7", "cam8"]
    ]

    # 创建状态帧（用于显示连接状态）
    def create_status_frame(text, width, height, color=(0, 0, 0)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 800  # 根据屏幕大小调整字体比例
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    while not stopped:
        try:
            # 创建画布
            canvas = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)

            # 获取所有流的帧和状态
            frames = {}
            stats = {}
            if stream_manager:
                # 获取处理后的帧（带检测结果）
                frames = stream_manager.get_all_frames(processed=True)
                stats = stream_manager.get_all_stats()

            # 填充每个位置
            for row_idx, row in enumerate(layout):
                for col_idx, stream_id in enumerate(row):
                    # 计算位置
                    x_start = col_idx * SMALL_SCREEN_WIDTH
                    y_start = row_idx * SMALL_SCREEN_HEIGHT

                    # 获取帧或创建状态帧
                    if stream_id in frames and frames[stream_id] is not None:
                        # 有有效帧，调整大小并显示
                        frame = cv2.resize(frames[stream_id], (SMALL_SCREEN_WIDTH, SMALL_SCREEN_HEIGHT))
                        # 添加流ID和状态信息
                        status_text = f"{stream_id}"
                        if stream_id in stats:
                            status = stats[stream_id].get("connection_status", "unknown")
                            # 截断状态文本，避免太长
                            if len(status) > 20:
                                status = status[:20] + "..."
                            status_text += f" - {status}"

                            # 添加检测状态
                            detection_enabled = stats[stream_id].get("detection_enabled", False)
                            if detection_enabled:
                                status_text += " [DET]"

                            # 添加抽帧信息
                            frame_skip = stats[stream_id].get("frame_skip", 0)
                            if frame_skip > -1:
                                status_text += f" [SKIP:{frame_skip}]"

                        cv2.putText(frame, status_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # 没有有效帧，显示状态信息
                        if stream_id in stats:
                            status = stats[stream_id].get("connection_status", "disconnected")
                            if "error" in status or "disconnected" in status:
                                frame = create_status_frame(
                                    f"{stream_id}: {status}",
                                    SMALL_SCREEN_WIDTH,
                                    SMALL_SCREEN_HEIGHT,
                                    (0, 0, 100)  # 红色背景表示错误
                                )
                            elif "connecting" in status or "reconnecting" in status:
                                frame = create_status_frame(
                                    f"{stream_id}: {status}",
                                    SMALL_SCREEN_WIDTH,
                                    SMALL_SCREEN_HEIGHT,
                                    (0, 100, 100)  # 黄色背景表示连接中
                                )
                            else:
                                frame = create_status_frame(
                                    f"{stream_id}: {status}",
                                    SMALL_SCREEN_WIDTH,
                                    SMALL_SCREEN_HEIGHT,
                                    (0, 0, 0)  # 黑色背景表示未知状态
                                )
                        else:
                            frame = create_status_frame(
                                f"{stream_id}: Not configured",
                                SMALL_SCREEN_WIDTH,
                                SMALL_SCREEN_HEIGHT,
                                (50, 50, 50)  # 灰色背景表示未配置
                            )

                    # 将帧放置到画布上
                    canvas[y_start:y_start + SMALL_SCREEN_HEIGHT, x_start:x_start + SMALL_SCREEN_WIDTH] = frame

            # 显示画布
            cv2.imshow("Multi-Stream Display", canvas)

            # 检查按键
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                stopped = True
                break
            elif key == ord('d'):
                # 切换所有流的检测状态
                if stream_manager:
                    for stream_id in stream_manager.get_active_streams():
                        current = stream_manager.get_stream_stats(stream_id).get("detection_enabled", False)
                        stream_manager.enable_detection(stream_id, not current)
            elif key == ord('o'):
                # 切换到原始流（不进行推理）
                if stream_manager:
                    for stream_id in stream_manager.get_active_streams():
                        stream_manager.switch_to_original_stream(stream_id)
            elif key == ord('p'):
                # 切换到处理后的流（进行推理）
                if stream_manager:
                    for stream_id in stream_manager.get_active_streams():
                        stream_manager.switch_to_processed_stream(stream_id)

        except Exception as e:
            logger.error(f"Error in display loop: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()


# FastAPI路由
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global stream_manager, display_thread
    stream_manager = StreamManager()
    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()
    logger.info("Multi-stream display system started")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global stopped, stream_manager
    stopped = True
    if stream_manager:
        stream_manager.stop_all()
    logger.info("Multi-stream display system stopped")


@app.get("/")
async def root():
    """根路由，返回系统信息"""
    return {"message": "Multi-Stream Video Management System", "status": "running"}


@app.get("/streams")
async def list_streams():
    """获取所有流列表"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    streams = stream_manager.get_active_streams()
    stats = stream_manager.get_all_stats()

    return {
        "streams": streams,
        "stats": stats
    }


@app.post("/streams/{stream_id}")
async def add_stream(stream_id: str, config: StreamConfig):
    """添加新流"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.add_stream(stream_id, config.stream_url,
                                 buffer_size=config.buffer_size,
                                 hw_accel=config.hw_accel,
                                 timeout=config.timeout,
                                 reconnect_delay=config.reconnect_delay,
                                 max_retries=config.max_retries,
                                 keyframe_only=config.keyframe_only,
                                 frame_skip=config.frame_skip):
        return {"message": f"Stream {stream_id} added successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Stream {stream_id} already exists")


@app.delete("/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """删除流"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.remove_stream(stream_id):
        return {"message": f"Stream {stream_id} removed successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")


@app.post("/streams/{stream_id}/start")
async def start_stream(stream_id: str):
    """启动指定流"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.start_stream(stream_id):
        return {"message": f"Stream {stream_id} started successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")


@app.post("/streams/{stream_id}/stop")
async def stop_stream(stream_id: str):
    """停止指定流"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.stop_stream(stream_id):
        return {"message": f"Stream {stream_id} stopped successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")


@app.put("/streams/{stream_id}/settings")
async def update_stream_settings(stream_id: str, settings: StreamUpdate):
    """更新流设置"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    # 过滤掉None值
    update_params = {k: v for k, v in settings.model_dump().items() if v is not None}

    if not update_params:
        raise HTTPException(status_code=400, detail="No valid parameters provided for update")

    stream_manager.update_stream_settings(stream_id, **update_params)
    return {"message": f"Stream {stream_id} settings updated successfully"}


@app.get("/streams/{stream_id}/frame")
async def get_stream_frame(stream_id: str, processed: bool = True):
    """获取指定流的当前帧（JPEG格式）"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    frame = stream_manager.get_frame(stream_id, processed)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"No frame available for stream {stream_id}")

    # 将帧编码为JPEG
    _, jpeg_frame = cv2.imencode('.jpg', frame)

    return StreamingResponse(
        iter([jpeg_frame.tobytes()]),
        media_type="image/jpeg"
    )


@app.post("/streams/{stream_id}/detection")
async def toggle_detection(stream_id: str, enable: bool):
    """启用或禁用目标检测"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.enable_detection(stream_id, enable):
        status = "enabled" if enable else "disabled"
        return {"message": f"Detection {status} for stream {stream_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")


@app.post("/models")
async def load_model(config: DetectionConfig):
    """加载TensorRT模型"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    try:
        model_name = stream_manager.load_model(config)
        return {"message": f"Model {model_name} loaded successfully", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/models")
async def list_models():
    """获取已加载的模型列表"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    models = stream_manager.get_loaded_models()
    return {"models": models}


@app.post("/streams/{stream_id}/models/{model_name}")
async def add_model_to_stream(stream_id: str, model_name: str):
    """为视频流添加模型"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    try:
        stream_manager.add_model_to_stream(stream_id, model_name)
        return {"message": f"Model {model_name} added to stream {stream_id}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/streams/{stream_id}/models/{model_name}")
async def remove_model_from_stream(stream_id: str, model_name: str):
    """从视频流移除模型"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    stream_manager.remove_model_from_stream(stream_id, model_name)
    return {"message": f"Model {model_name} removed from stream {stream_id}"}


@app.get("/streams/{stream_id}/models")
async def get_stream_models(stream_id: str):
    """获取视频流激活的模型列表"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    models = stream_manager.get_stream_models(stream_id)
    return {"stream_id": stream_id, "models": models}


@app.post("/models/{model_name}/alerts")
async def add_alert_condition(model_name: str, condition: AlertCondition, action: AlertAction):
    """为模型添加报警条件"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    try:
        stream_manager.add_alert_condition(model_name, condition, action)
        return {"message": f"Alert condition added for model {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/streams/{stream_id}/switch/original")
async def switch_to_original_stream(stream_id: str):
    """切换到原始流（不进行推理）"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.switch_to_original_stream(stream_id):
        return {"message": f"Switched to original stream for {stream_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")


@app.post("/streams/{stream_id}/switch/processed")
async def switch_to_processed_stream(stream_id: str):
    """切换到处理后的流（进行推理）"""
    if not stream_manager:
        raise HTTPException(status_code=500, detail="Stream manager not initialized")

    if stream_manager.switch_to_processed_stream(stream_id):
        return {"message": f"Switched to processed stream for {stream_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")


if __name__ == "__main__":
    # 启动FastAPI应用
    uvicorn.run(app, host="0.0.0.0", port=8000)