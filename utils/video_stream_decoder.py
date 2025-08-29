import av
import cv2
import time
import logging
import threading
import numpy as np
from typing import Optional, Any, Dict, List, Tuple, Callable
import datetime
from collections import defaultdict

logger = logging.getLogger("StreamDecoder")

def add_timestamp(frame_array):
    """添加时间戳到视频帧"""
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    font_thickness = 2
    background_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(time_str, font, font_scale, font_thickness)
    margin = 10
    text_x = frame_array.shape[1] - text_width - margin
    text_y = text_height + margin

    bg_rect = np.zeros((text_height + 2 * margin, text_width + 2 * margin, 3), dtype=np.uint8)
    bg_rect[:, :] = background_color

    alpha = 0.6
    y1, y2 = text_y - text_height - margin, text_y + margin
    x1, x2 = text_x - margin, text_x + text_width + margin

    y1, y2 = max(0, y1), min(frame_array.shape[0], y2)
    x1, x2 = max(0, x1), min(frame_array.shape[1], x2)

    if y2 > y1 and x2 > x1:
        roi = frame_array[y1:y2, x1:x2]
        bg_resized = cv2.resize(bg_rect, (x2 - x1, y2 - y1))
        blended = cv2.addWeighted(roi, 1 - alpha, bg_resized, alpha, 0)
        frame_array[y1:y2, x1:x2] = blended

    cv2.putText(
        frame_array,
        time_str,
        (text_x, text_y),
        font,
        font_scale,
        font_color,
        font_thickness,
        cv2.LINE_AA
    )

    return frame_array


class VideoStreamDecoder:
    def __init__(self, stream_id: str, stream_url: str, buffer_size: int = 102400,
                 hw_accel: Optional[str] = None, timeout: int = 5000000,
                 reconnect_delay: int = 5, max_retries: int = -1,
                 keyframe_only: bool = False, frame_skip: int = 3,
                 inference_callback: Optional[Callable] = None):
        self.stream_id = stream_id
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.hw_accel = hw_accel
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.keyframe_only = keyframe_only
        self.frame_skip = frame_skip
        self.inference_callback = inference_callback
        self.detection_enabled = False

        self._retry_count = 0
        self._connection_status = "disconnected"  # disconnected, connecting, connected, error
        self._frame_counter = 0

        self._container: Optional[av.container.InputContainer] = None
        self._video_stream: Optional[av.video.stream.VideoStream] = None
        self._current_iterator: Optional[Any] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_processed_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._keyframe_count: int = 0
        self._running = False
        self._decode_thread: Optional[threading.Thread] = None
        self._options = self._get_codec_options()
        self.lock = threading.Lock()

    def _get_codec_options(self) -> Dict:
        """根据硬件加速类型获取解码器选项"""
        options = {'threads': 'auto'}
        if self.hw_accel:
            options.update({
                'hwaccel': self.hw_accel,
            })
        return options

    def connect(self) -> bool:
        """尝试连接流并初始化视频流"""
        try:
            self._connection_status = "connecting"
            self._container = av.open(
                self.stream_url,
                options={
                    'rtsp_flags': 'prefer_tcp',
                    'buffer_size': str(self.buffer_size),
                    'stimeout': str(self.timeout),
                },
                timeout=(self.timeout / 1000000)
            )

            self._video_stream = next(s for s in self._container.streams if s.type == 'video')

            if self.keyframe_only:
                self._video_stream.codec_context.skip_frame = 'NONKEY'

            self._current_iterator = self._container.decode(self._video_stream)
            logger.info(f"Successfully connected to {self.stream_url}")
            self._retry_count = 0
            self._connection_status = "connected"
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.stream_url}: {e}")
            self._connection_status = f"error: {str(e)}"
            return False

    def _process_frame_callback(self, processed_frame: np.ndarray, detections: List[Dict]):
        """处理完成后的回调函数"""
        with self.lock:
            self._last_processed_frame = processed_frame

    def _decode_loop(self):
        """运行在独立线程中的解码循环"""
        while self._running:
            try:
                frame = next(self._current_iterator)
                self._frame_count += 1
                if frame.key_frame:
                    self._keyframe_count += 1

                frame_array = frame.to_ndarray(format='bgr24')
                frame_array = add_timestamp(frame_array)

                with self.lock:
                    self._last_frame = frame_array

                # 处理检测 - 异步方式
                if self.detection_enabled and self.inference_callback:
                    # 抽帧逻辑：每 frame_skip+1 帧处理一帧
                    if self._frame_counter % (self.frame_skip + 1) == 0:
                        self.inference_callback(self.stream_id, frame_array, self._process_frame_callback)
                    self._frame_counter += 1
                else:
                    # 如果没有启用检测，确保处理后的帧为空
                    with self.lock:
                        self._last_processed_frame = None

            except (av.AVError, StopIteration, ValueError) as e:
                logger.warning(f"Decoding error on {self.stream_url}: {e}. Attempting reconnect...")
                self._connection_status = f"reconnecting: {str(e)}"
                if self._container:
                    self._container.close()
                if not self._attempt_reconnect():
                    continue
            except Exception as e:
                logger.error(f"Unexpected error in decode loop for {self.stream_url}: {e}")
                self._connection_status = f"error: {str(e)}"
                with self.lock:
                    self._last_frame = None
                    self._last_processed_frame = None
                time.sleep(1)

    def _attempt_reconnect(self) -> bool:
        """尝试重连，根据重连策略"""
        if self.max_retries > 0 and self._retry_count >= self.max_retries:
            logger.error(f"Reached max retries ({self.max_retries}) for {self.stream_url}. Giving up.")
            self._connection_status = "disconnected"
            return False

        self._retry_count += 1
        logger.info(f"Attempting to reconnect ({self._retry_count}) in {self.reconnect_delay} seconds...")
        time.sleep(self.reconnect_delay)

        if self.connect():
            return True
        return False

    def start(self):
        """启动解码线程"""
        if self._running:
            logger.warning(f"Decoder for {self.stream_url} is already running.")
            return

        if self._container is None:
            if not self.connect():
                logger.error(f"Failed to start because connection failed for {self.stream_url}.")
                return

        self._running = True
        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._decode_thread.start()
        logger.info(f"Started decoder thread for {self.stream_url}.")

    def get_frame(self, processed: bool = False) -> Optional[np.ndarray]:
        """获取当前最新的视频帧（numpy array）"""
        with self.lock:
            if processed and self.detection_enabled and self._last_processed_frame is not None:
                return self._last_processed_frame.copy()
            elif self._last_frame is not None:
                return self._last_frame.copy()
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取解码统计信息"""
        return {
            "frame_count": self._frame_count,
            "keyframe_count": self._keyframe_count,
            "retry_count": self._retry_count,
            "connection_status": self._connection_status,
            "detection_enabled": self.detection_enabled,
            "frame_skip": self.frame_skip
        }

    def update_settings(self, **kwargs):
        """动态更新参数"""
        allowed_params = ['buffer_size', 'timeout', 'reconnect_delay', 'max_retries', 'keyframe_only', 'frame_skip']
        need_restart = False

        for key, value in kwargs.items():
            if key in allowed_params and hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated {key} from {old_value} to {value} for {self.stream_url}.")

                if key in ['keyframe_only', 'timeout', 'buffer_size']:
                    need_restart = True

        if need_restart and self._running:
            self.restart()

    def enable_detection(self, enable: bool):
        """启用或禁用目标检测"""
        self.detection_enabled = enable
        # 重置帧计数器
        self._frame_counter = 0
        logger.info(f"{'Enabled' if enable else 'Disabled'} detection for {self.stream_url}")

    def stop(self):
        """停止解码并清理资源"""
        self._running = False
        self._connection_status = "disconnected"
        if self._decode_thread and self._decode_thread.is_alive():
            self._decode_thread.join(timeout=2.0)
        if self._container:
            self._container.close()
        logger.info(f"Stopped decoder for {self.stream_url}.")

    def restart(self):
        """重启流"""
        self.stop()
        time.sleep(1)
        self._container = None
        self._video_stream = None
        self.start()