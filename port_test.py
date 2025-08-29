import os.path

import requests
import json
import time


class VideoStreamController:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def _make_request(self, method, endpoint, data=None):
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PUT":
                response = self.session.put(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            if response.status_code >= 200 and response.status_code < 300:
                return response.json()
            else:
                print(f"请求失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"请求出错: {e}")
            return None

    def add_stream(self, stream_id, stream_url, **kwargs):
        """添加视频流"""
        data = {
            "stream_url": stream_url,
            "buffer_size": 5,
            "hw_accel": kwargs.get("hw_accel", None),
            "timeout": kwargs.get("timeout", 5000000),
            "reconnect_delay": kwargs.get("reconnect_delay", 5),
            "max_retries": kwargs.get("max_retries", -1),
            "keyframe_only": kwargs.get("keyframe_only", False)
        }
        return self._make_request("POST", f"/streams/{stream_id}", data)

    def start_stream(self, stream_id):
        """启动视频流"""
        return self._make_request("POST", f"/streams/{stream_id}/start")

    def stop_stream(self, stream_id):
        """停止视频流"""
        return self._make_request("POST", f"/streams/{stream_id}/stop")

    def remove_stream(self, stream_id):
        """移除视频流"""
        return self._make_request("DELETE", f"/streams/{stream_id}")

    def list_streams(self):
        """获取所有流列表"""
        return self._make_request("GET", "/streams")

    def update_stream_settings(self, stream_id, **kwargs):
        """更新流设置"""
        data = {}
        allowed_params = ["buffer_size", "hw_accel", "timeout", "reconnect_delay", "max_retries", "keyframe_only","frame_skip"]

        for key, value in kwargs.items():
            if key in allowed_params:
                data[key] = value

        return self._make_request("PUT", f"/streams/{stream_id}/settings", data)

    def get_frame(self, stream_id, processed=True):
        """获取视频帧（返回字节数据）"""
        url = f"{self.base_url}/streams/{stream_id}/frame?processed={str(processed).lower()}"
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                return response.content
            else:
                print(f"获取帧失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"请求出错: {e}")
            return None

    def toggle_detection(self, stream_id, enable):
        """启用或禁用目标检测"""
        return self._make_request("POST", f"/streams/{stream_id}/detection?enable={str(enable).lower()}")

    def load_model(self, model_path, confidence_threshold=0.5, classes=None):
        """加载YOLOv8模型"""
        data = {
            "model_path": r"weights/"+str(model_path),
            "confidence_threshold": confidence_threshold,
            "classes": classes
        }
        return self._make_request("POST", "/models", data)

    def list_models(self):
        """获取已加载的模型列表"""
        return self._make_request("GET", "/models")

    def add_model_to_stream(self, stream_id, model_name):
        """为视频流添加模型"""
        return self._make_request("POST", f"/streams/{stream_id}/models/{model_name}")

    def remove_model_from_stream(self, stream_id, model_name):
        """从视频流移除模型"""
        return self._make_request("DELETE", f"/streams/{stream_id}/models/{model_name}")

    def get_stream_models(self, stream_id):
        """获取视频流激活的模型列表"""
        return self._make_request("GET", f"/streams/{stream_id}/models")

    def add_alert_condition(self, model_name, class_name, min_confidence=0.5, min_count=1,
                            max_count=None, cooldown=5, action_type="log", action_config=None):
        """为模型添加报警条件"""
        if action_config is None:
            action_config = {}

        condition = {
            "class_name": class_name,
            "min_confidence": min_confidence,
            "min_count": min_count,
            "max_count": max_count,
            "cooldown": cooldown
        }

        action = {
            "action_type": action_type,
            "config": action_config
        }

        data = {
            "condition": condition,
            "action": action
        }

        return self._make_request("POST", f"/models/{model_name}/alerts", data)


# 使用示例
if __name__ == "__main__":
    controller = VideoStreamController()

    #---------------------------------------------------------------
    ## 1. 添加视频流
    print("1. 添加视频流")
    controller.add_stream("cam1", "rtsp://localhost:5001/stream_1")
    controller.add_stream("cam2", "rtsp://localhost:5001/stream_2")
    controller.add_stream("cam3", "rtsp://localhost:5001/stream_3")
    controller.add_stream("cam4", "rtsp://localhost:5001/stream_4")

    controller.add_stream("cam5", "rtsp://localhost:5001/stream_1")
    controller.add_stream("cam6", "rtsp://localhost:5001/stream_2")
    controller.add_stream("cam7", "rtsp://localhost:5001/stream_3")
    controller.add_stream("cam8", "rtsp://localhost:5001/stream_4")
    #
    # # 2. 启动视频流
    # print("2. 启动视频流")
    controller.start_stream("cam1")
    controller.start_stream("cam2")
    controller.start_stream("cam3")
    controller.start_stream("cam4")

    controller.start_stream("cam5")
    controller.start_stream("cam6")
    controller.start_stream("cam7")
    controller.start_stream("cam8")
    # ---------------------------------------------------

    time.sleep(3)

    #--------------------------------------------------------
    # # 3. 加载YOLOv8模型
    print("3. 加载YOLOv8模型")
    result = controller.load_model("yolo11n_fp16_final.engine", confidence_threshold=0.5, classes=[0, 1, 2])
    if result:
        model_name = result.get("model_name")
        print(f"已加载模型: {model_name}")
    # time.sleep(2)
    # # 4. 为视频流添加模型
    # print("4. 为视频流添加模型")
    controller.add_model_to_stream("cam1", model_name)
    controller.add_model_to_stream("cam2", model_name)
    controller.add_model_to_stream("cam3", model_name)
    controller.add_model_to_stream("cam4", model_name)
    controller.add_model_to_stream("cam5", model_name)
    controller.add_model_to_stream("cam6", model_name)
    controller.add_model_to_stream("cam7", model_name)
    controller.add_model_to_stream("cam8", model_name)



    # 5. 启用检测
    print("5. 启用检测")
    controller.toggle_detection("cam1", True)
    controller.toggle_detection("cam2", True)
    controller.toggle_detection("cam3", True)
    controller.toggle_detection("cam4", True)
    controller.toggle_detection("cam5", True)
    controller.toggle_detection("cam6", True)
    controller.toggle_detection("cam7", True)
    controller.toggle_detection("cam8", True)

    time.sleep(10)
    controller.update_stream_settings("cam1", frame_skip=0)
    controller.update_stream_settings("cam2", frame_skip=0)
    controller.update_stream_settings("cam3", frame_skip=0)
    controller.update_stream_settings("cam4", frame_skip=0)
    controller.update_stream_settings("cam5", frame_skip=0)
    controller.update_stream_settings("cam6", frame_skip=0)
    controller.update_stream_settings("cam7", frame_skip=0)
    controller.update_stream_settings("cam8", frame_skip=0)

    #    ---------------------------------------------------------
    # 6. 添加报警条件
    # print("6. 添加报警条件")
    # controller.add_alert_condition(
    #     model_name,
    #     "person",
    #     min_confidence=0.7,
    #     min_count=1,
    #     action_type="log"
    # )
    #
    # controller.add_alert_condition(
    #     model_name,
    #     "car",
    #     min_confidence=0.6,
    #     min_count=2,
    #     action_type="log"
    # )
    #
    # # 7. 获取带检测结果的帧
    # print("7. 获取带检测结果的帧")
    # frame_data = controller.get_frame("cam1", processed=True)
    # if frame_data:
    #     with open("detected_frame.jpg", "wb") as f:
    #         f.write(frame_data)
    #     print("已保存带检测结果的帧为 detected_frame.jpg")
    #
    # # 8. 查看当前状态
    # print("8. 查看当前状态")
    # streams = controller.list_streams()
    # if streams:
    #     print("当前所有流状态:")
    #     print(json.dumps(streams, indent=2, ensure_ascii=False))
    #
    # models = controller.list_models()
    # if models:
    #     print("已加载的模型:")
    #     print(json.dumps(models, indent=2, ensure_ascii=False))
    # time.sleep(10)


    # # 9. 查看特定流的模型
    # cam1_models = controller.get_stream_models("cam1")
    # if cam1_models:
    #     print("cam1激活的模型:")
    #     print(json.dumps(cam1_models, indent=2, ensure_ascii=False))
    #
    # # 10. 停止检测
    # print("10. 停止检测")
    time.sleep(10)
    controller.toggle_detection("cam1", False)
    controller.toggle_detection("cam2", False)
    controller.toggle_detection("cam3", False)
    controller.toggle_detection("cam4", False)
    controller.toggle_detection("cam5", False)
    controller.toggle_detection("cam6", False)
    controller.toggle_detection("cam7", False)
    controller.toggle_detection("cam8", False)
    time.sleep(5)

    #
    # # 11. 从流中移除模型
    # print("11. 从流中移除模型")
    controller.remove_model_from_stream("cam1", model_name)
    #
    # # 12. 停止并移除所有流
    # print("12. 停止并移除所有流")
    controller.stop_stream("cam1")
    controller.stop_stream("cam2")
    controller.stop_stream("cam3")
    controller.stop_stream("cam4")
    controller.stop_stream("cam5")
    controller.stop_stream("cam6")
    controller.stop_stream("cam7")
    controller.stop_stream("cam8")
    time.sleep(1)
    controller.remove_stream("cam1")
    controller.remove_stream("cam2")
    controller.remove_stream("cam3")
    controller.remove_stream("cam4")
    controller.remove_stream("cam5")
    controller.remove_stream("cam6")
    controller.remove_stream("cam7")
    controller.remove_stream("cam8")

    # controller.update_stream_settings("cam1", keyframe_only=False,frame_skip = 0)
    print("演示完成!")