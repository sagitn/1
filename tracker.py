import cv2
import numpy as np
import time
import math
import queue
import threading
import serial
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import bettercam
import torch
import torchvision
import tensorrt as trt


# ==========================================================
# 安全版实时目标跟踪与相机视角跟随脚本
# ----------------------------------------------------------
# 这份脚本的定位：
# 1) 使用 YOLO(TensorRT) 做实时检测
# 2) 使用 Kalman + 简单数据关联做多目标跟踪
# 3) 使用“目标级别”的类别时序平滑，避免单帧标签闪烁
# 4) 选择一个稳定目标，对外输出“相机/云台视角跟随”控制量
#
# 为什么要拆成四层：
# - detector(检测)   ：只负责“看见什么”
# - tracker(跟踪)    ：只负责“这个目标是不是上一帧那个目标”
# - selector(选择)   ：只负责“当前该跟随哪一个目标”
# - controller(控制) ：只负责“如何把像素误差变成平滑控制量”
#
# 这样拆分的好处：
# - 更容易调试：可以分别判断是检测不准，还是跟踪不稳，还是控制过冲
# - 更容易迁移：后续换模型、换串口协议、换控制设备时，不必重写整个脚本
# - 更适合课程展示：模块边界清晰，方便解释每层的职责
#
# 注意：
# 本脚本输出的是“相机/云台视角跟随”的串口增量，便于连接外部控制设备做目标跟踪录制。
# ==========================================================


# =============================
# 1) 配置区
# =============================
@dataclass
class AppConfig:
    # -------------------------
    # 图像采集相关配置
    # -------------------------
    # region: 截图区域 (left, top, right, bottom)
    # 为什么只截取局部区域：
    # - 可以减少输入分辨率，明显降低 3060 上的推理压力
    # - 你的场景如果关注中心区域，裁剪能直接提升帧率
    region: Tuple[int, int, int, int] = (768, 208, 1792, 1232)

    # imgsz: 模型输入尺寸
    # 为什么不是越大越好：
    # - 更大输入会带来更高检测精度，但也会大幅增加推理延迟
    # - 对 30~50 FPS 的环境，通常要优先保持实时性
    imgsz: int = 896

    show_window: bool = True
    output_idx: int = 0

    # -------------------------
    # TensorRT / YOLO 相关配置
    # -------------------------
    engine_path: str = r"D:/models/your_model.engine"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_det: int = 50

    # -------------------------
    # 跟踪相关配置
    # -------------------------
    # max_lost_frames:
    # 一个 track 连续多少帧没匹配到检测框就删除
    # 为什么要允许短暂丢失：
    # - 检测器偶尔漏检是正常现象
    # - 如果一漏检就删除，ID 会频繁重建，轨迹会很抖
    max_lost_frames: int = 18

    # min_confirmed_hits:
    # 新目标至少连续命中多少次才视为“确认目标”
    # 为什么这样做：
    # - 可以过滤偶发误检
    # - 避免刚出现一帧的杂点就被选为当前目标
    min_confirmed_hits: int = 2

    # association_dist_px:
    # 数据关联时，检测框与预测位置的最大允许距离
    association_dist_px: float = 140.0

    # association_iou_min:
    # 数据关联时，至少保留一点重叠约束，避免离得近但实际上不是同一目标
    association_iou_min: float = 0.05

    # track_history:
    # 每个轨迹保留多少个历史点，用于画轨迹和观察平滑程度
    track_history: int = 24

    # class_smooth_alpha:
    # 类别平滑中的历史保留系数
    # 越接近 1，越保守，类别更稳定但响应更慢
    class_smooth_alpha: float = 0.75

    # -------------------------
    # 目标选择相关配置
    # -------------------------
    # class_allowlist:
    # 只允许跟随哪些类别
    # 例如只跟随 person，不跟随 friend / other
    class_allowlist: Tuple[str, ...] = ("person",)
    class_blocklist: Tuple[str, ...] = ()

    # target_stickiness_bonus:
    # 对“当前已经锁定的目标”给予一个粘性加分
    # 为什么需要粘性：
    # - 多目标靠近时，如果每帧都只看当前分数最高者，目标会频繁切换
    # - 粘性能让系统更愿意继续跟随已经跟随中的目标
    target_stickiness_bonus: float = 120.0

    # lost_target_grace_s:
    # 当前目标短暂丢失后，允许保留多长时间
    # 为什么需要这个宽限：
    # - 遮挡、漏检时如果马上切到别的目标，会非常抖
    lost_target_grace_s: float = 0.25

    # -------------------------
    # 控制器相关配置（用于相机/云台视角跟随）
    # -------------------------
    control_hz: float = 120.0

    # 这里使用 PD 控制，而不是完整 PID
    # 为什么默认不加 I（积分项）：
    # - 实时视觉系统本身有检测延迟和帧率波动
    # - I 项很容易积累过量，导致明显过冲和来回摆动
    kp_x: float = 0.018
    kd_x: float = 0.006
    kp_y: float = 0.018
    kd_y: float = 0.006

    # deadzone_px:
    # 目标已经接近画面中心时，不再输出微小控制量
    # 为什么需要死区：
    # - 可以减少“已经对准但还在抖”的现象
    deadzone_px: float = 6.0

    # max_delta_per_tick:
    # 单个控制周期内允许的最大输出
    # 为什么要限幅：
    # - 防止误差突然变大时输出过猛
    # - 防止目标切换时出现剧烈跳动
    max_delta_per_tick: float = 35.0

    # 这里保留积分上限字段，但默认关闭积分
    max_integrator: float = 0.0

    # command_timeout_s:
    # 如果控制命令超过这个时间没刷新，就自动发 0
    # 为什么要做超时保护：
    # - 避免图像线程卡住时，外设还在沿着旧命令继续转动
    command_timeout_s: float = 0.10

    # -------------------------
    # 串口输出配置
    # -------------------------
    serial_enabled: bool = True
    com_port: str = "COM11"
    baud_rate: int = 921600

    # -------------------------
    # 热键配置（Windows）
    # -------------------------
    vk_f9: int = 0x78
    vk_end: int = 0x23

    # -------------------------
    # 类别名配置
    # -------------------------
    # 这里要与你自己的模型类别顺序保持一致
    class_names: Tuple[str, ...] = ("person",)


# =============================
# 2) 基础工具函数
# =============================
def now() -> float:
    """返回高精度计时器，适合做实时系统的 dt 计算。"""
    return time.perf_counter()


def clamp(v: float, lo: float, hi: float) -> float:
    """把数值限制在 [lo, hi] 范围内，常用于控制输出限幅。"""
    return max(lo, min(hi, v))


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    把 [cx, cy, w, h] 转成 [x1, y1, x2, y2]。

    为什么要转：
    - NMS 常用 xyxy 格式
    - 计算 IoU 时也更直接
    """
    out = torch.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个框的 IoU。

    为什么在关联里还要用 IoU：
    - 只看中心点距离时，两个目标靠近交叉时容易串 ID
    - 加一点 IoU 约束能让“看起来像同一目标”的判断更可靠
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


# =============================
# 3) 数据结构
# =============================
@dataclass
class Detection:
    """
    单帧检测结果。

    为什么单独封装 Detection：
    - 便于 detector 和 tracker 解耦
    - 后续若切换模型，只要最终能组装成 Detection 即可
    """
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    class_id: int
    class_name: str
    score_vec: Optional[np.ndarray] = None

    @property
    def xyxy(self) -> np.ndarray:
        """按需把检测框转换成 xyxy。"""
        return np.array([
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        ], dtype=np.float32)


@dataclass
class SmoothedClassState:
    """
    类别时序平滑器。

    作用：
    - 不再只相信当前这一帧的类别
    - 把历史类别概率保留下来，降低标签闪烁

    为什么需要它：
    - 现实里单帧分类经常会在 friend / person / other 之间抖动
    - 如果每帧都直接用当前类别，目标选择会非常不稳定
    """
    alpha: float
    probs: Dict[int, float] = field(default_factory=dict)
    stable_class_id: int = -1

    def update(self, class_id: int, conf: float):
        """
        用指数平滑方式更新类别概率。

        做法：
        - 旧概率先乘 alpha，表示“保留历史记忆”
        - 当前类别加上 (1-alpha)*conf，表示“吸收当前帧证据”
        - 最后重新归一化
        """
        new_probs: Dict[int, float] = {}
        for k, v in self.probs.items():
            new_probs[k] = v * self.alpha
        new_probs[class_id] = new_probs.get(class_id, 0.0) + (1.0 - self.alpha) * conf
        total = sum(new_probs.values())
        if total > 1e-6:
            for k in list(new_probs.keys()):
                new_probs[k] /= total
        self.probs = new_probs
        self.stable_class_id = max(self.probs, key=self.probs.get)

    def confidence(self) -> float:
        """返回当前稳定类别的平滑后置信度。"""
        if self.stable_class_id < 0:
            return 0.0
        return self.probs.get(self.stable_class_id, 0.0)


class Kalman2D:
    """
    二维常速度 Kalman 滤波器，状态量为 [x, y, vx, vy]。

    为什么这里用 Kalman：
    - 检测器给的是“离散的、带噪声的位置测量”
    - Kalman 可以同时估计位置和速度
    - 当出现漏检时，仍然能短时间预测目标会出现在哪里

    为什么选择常速度模型：
    - 计算量小，适合 30~50 FPS 的实时环境
    - 对多数横移/走动目标已经足够有效
    - 比加入加速度的模型更简单、稳定、好调
    """
    def __init__(self, x: float, y: float):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 8.0
        self.last_ts = now()

    def predict(self, dt: float):
        """
        预测步骤。

        为什么必须带 dt：
        - 你的环境不是固定 120 FPS，而是 30~50 FPS 波动
        - 如果不把 dt 带进去，帧率一变，预测就会明显失真
        """
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Q 是过程噪声。
        # 这里做一个简单的 dt 相关噪声模型，避免 dt 太小或太大时状态发散。
        q = max(dt, 1 / 240.0)
        Q = np.array([
            [q*q, 0, q, 0],
            [0, q*q, 0, q],
            [q, 0, 1, 0],
            [0, q, 0, 1],
        ], dtype=np.float32) * 3.0

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_x: float, z_y: float):
        """
        更新步骤：用当前检测到的位置修正预测结果。
        """
        z = np.array([[z_x], [z_y]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        """返回当前估计位置。"""
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def vel(self) -> Tuple[float, float]:
        """返回当前估计速度。"""
        return float(self.x[2, 0]), float(self.x[3, 0])


@dataclass
class Track:
    """
    一个持续存在的目标轨迹。

    为什么需要 Track 而不只存当前框：
    - 目标跟踪的核心不是“这一帧框在哪”，而是“这是不是同一个目标”
    - Track 里会保存：预测器、历史类别、历史位置、丢失次数等
    - 只有这样才能做“目标级别决策”，而不是“框级别决策”
    """
    track_id: int
    kf: Kalman2D
    smooth_class: SmoothedClassState
    last_box: np.ndarray
    last_conf: float
    hits: int = 1
    lost_frames: int = 0
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    history: deque = field(default_factory=lambda: deque(maxlen=24))

    def predict(self, dt: float):
        """让该轨迹先进行一步预测。"""
        self.kf.predict(dt)

    def update(self, det: Detection):
        """
        用新检测结果更新轨迹。

        这里同时做了三件事：
        1) 更新 Kalman 位置估计
        2) 更新最后一个框与置信度
        3) 更新类别时序平滑结果
        """
        self.kf.update(det.cx, det.cy)
        self.last_box = det.xyxy.copy()
        self.last_conf = det.conf
        self.hits += 1
        self.lost_frames = 0
        self.updated_at = now()
        self.smooth_class.update(det.class_id, det.conf)
        self.history.append((int(det.cx), int(det.cy)))

    def miss(self):
        """本帧没有匹配到检测框，记为一次丢失。"""
        self.lost_frames += 1

    @property
    def center(self) -> Tuple[float, float]:
        """返回 Kalman 估计的轨迹中心。"""
        return self.kf.pos

    @property
    def velocity(self) -> Tuple[float, float]:
        """返回 Kalman 估计的轨迹速度。"""
        return self.kf.vel

    @property
    def stable_class_id(self) -> int:
        """返回平滑后的稳定类别 ID。"""
        return self.smooth_class.stable_class_id

    def is_confirmed(self, min_hits: int) -> bool:
        """新目标达到一定命中次数后才视为有效轨迹。"""
        return self.hits >= min_hits


# =============================
# 4) 检测器：TensorRT 封装
# =============================
class TRTEngine:
    """
    低层 TensorRT 封装。

    为什么单独封装一层：
    - 把引擎加载、输入输出绑定和执行过程隔离出去
    - 以后换模型或换输出格式时，不影响上层 tracker / selector / controller
    """
    def __init__(self, engine_path: str, device=torch.device("cuda:0")):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.stream = torch.cuda.Stream(device=device)

        # 预先分配输入输出张量，避免每帧重复申请显存。
        # 为什么这么做：
        # - 频繁申请/释放显存会增加抖动和延迟
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = list(self.engine.get_tensor_shape(name))
            if shape[0] == -1:
                shape[0] = 1
                self.context.set_input_shape(name, tuple(shape))
            tensor = torch.empty(tuple(shape), dtype=torch.from_numpy(np.empty(0, dtype=dtype)).dtype, device=device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            slot = {"name": name, "tensor": tensor, "shape": tuple(shape)}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(slot)
            else:
                self.outputs.append(slot)

    def infer(self, img_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        执行一次推理。

        注意：
        这里已经假设 img_tensor 的尺寸和 dtype 与引擎要求匹配。
        """
        self.inputs[0]["tensor"].copy_(img_tensor)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return [o["tensor"] for o in self.outputs]


class DetectorTRT:
    """
    上层检测器封装。

    职责：
    - 预处理：缩放 / BGR->RGB / NCHW / 归一化
    - 调用 TensorRT 推理
    - 后处理：解析输出、阈值过滤、NMS、组装 Detection
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.engine = TRTEngine(cfg.engine_path, self.device)
        self.region_w = cfg.region[2] - cfg.region[0]
        self.region_h = cfg.region[3] - cfg.region[1]
        self.ratio_x = self.region_w / cfg.imgsz
        self.ratio_y = self.region_h / cfg.imgsz

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        图像预处理。

        为什么这里保持流程简单：
        - 预处理本身也会消耗 CPU
        - 在 3060 + 普通 CPU 环境下，简单稳定比花哨增强更重要
        """
        img = cv2.resize(frame_bgr, (self.cfg.imgsz, self.cfg.imgsz), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img).to(self.device, non_blocking=True).float().unsqueeze(0) / 255.0

    def postprocess_common_yolo(self, preds: torch.Tensor) -> List[Detection]:
        """
        按常见 YOLO 导出格式解析 TRT 输出。

        这里假设输出近似为：
        - [1, N, 4 + 1 + num_classes] 或
        - [1, 4 + 1 + num_classes, N]

        如果你的 engine 输出格式不同，需要改这个函数。
        这是最常见、也是最需要按你的实际模型微调的地方。
        """
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            preds = preds.squeeze(0).transpose(0, 1)
        else:
            preds = preds.squeeze(0)

        if preds.shape[1] < 6:
            raise RuntimeError("TRT output shape not recognized. Adjust postprocess_common_yolo().")

        boxes = preds[:, :4]
        obj = preds[:, 4:5]
        cls_scores = preds[:, 5:]

        # 如果没有类别分支，就默认全是 0 类。
        if cls_scores.numel() == 0:
            class_ids = torch.zeros((preds.shape[0],), dtype=torch.long, device=preds.device)
            confs = obj.squeeze(1)
        else:
            cls_conf, class_ids = torch.max(cls_scores, dim=1)
            confs = obj.squeeze(1) * cls_conf

        # 先做一次置信度过滤，减少后续 NMS 的计算量。
        mask = confs > self.cfg.conf_threshold
        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]
        cls_scores = cls_scores[mask] if cls_scores.numel() else cls_scores

        if boxes.numel() == 0:
            return []

        xyxy = xywh_to_xyxy(boxes)
        keep = torchvision.ops.nms(xyxy, confs, self.cfg.iou_threshold)
        keep = keep[: self.cfg.max_det]

        dets: List[Detection] = []
        for idx in keep.tolist():
            # 模型坐标系 -> 截图区域坐标系
            cx = float(boxes[idx, 0]) * self.ratio_x
            cy = float(boxes[idx, 1]) * self.ratio_y
            w = float(boxes[idx, 2]) * self.ratio_x
            h = float(boxes[idx, 3]) * self.ratio_y
            class_id = int(class_ids[idx])
            class_name = self.cfg.class_names[class_id] if class_id < len(self.cfg.class_names) else f"cls_{class_id}"
            score_vec = None
            if cls_scores.numel():
                score_vec = cls_scores[idx].detach().float().cpu().numpy()
            dets.append(Detection(cx=cx, cy=cy, w=w, h=h, conf=float(confs[idx]), class_id=class_id, class_name=class_name, score_vec=score_vec))
        return dets

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """执行完整检测流程。"""
        img_tensor = self.preprocess(frame_bgr)
        outputs = self.engine.infer(img_tensor)
        return self.postprocess_common_yolo(outputs[0])


# =============================
# 5) 多目标跟踪器
# =============================
class MultiObjectTracker:
    """
    多目标跟踪器。

    核心思路：
    1) 每一帧先让所有旧轨迹按 Kalman 预测到当前时刻
    2) 把检测结果与轨迹做匹配（数据关联）
    3) 匹配上的轨迹更新，没匹配上的轨迹记为丢失
    4) 没匹配上的检测生成新轨迹

    为什么这是关键优化点：
    - 单纯“找离上一帧最近的框”在多目标下非常容易切换抖动
    - 加入预测后，系统会更倾向于维持原来的目标身份
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.last_ts = now()

    def _association_cost(self, trk: Track, det: Detection) -> float:
        """
        计算某个轨迹与某个检测之间的关联代价。

        代价越小，越可能是同一个目标。

        这里使用：
        - 中心点距离
        - IoU 轻度约束

        为什么不用更复杂的 ReID 特征：
        - 你的环境强调实时性和部署简单
        - 30~50 FPS 下，先把经典方法做稳，往往收益更大
        """
        px, py = trk.center
        dist = math.hypot(det.cx - px, det.cy - py)
        iou = bbox_iou_xyxy(trk.last_box, det.xyxy)

        # 超过最大关联距离，直接判定不可能匹配。
        if dist > self.cfg.association_dist_px:
            return float("inf")

        # 如果 IoU 极低且距离也不够近，也认为不可靠。
        if iou < self.cfg.association_iou_min and dist > self.cfg.association_dist_px * 0.5:
            return float("inf")

        # 用“距离 - IoU奖励”构成简单代价。
        return dist - 40.0 * iou

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        用当前帧所有检测结果更新全部轨迹。
        """
        ts = now()
        dt = max(1 / 240.0, ts - self.last_ts)
        self.last_ts = ts

        # 第一步：所有轨迹先预测到当前时刻
        for trk in self.tracks.values():
            trk.predict(dt)

        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        pairs: List[Tuple[float, int, int]] = []

        # 构建所有可能的匹配对及其代价
        for tid, trk in self.tracks.items():
            for did, det in enumerate(detections):
                cost = self._association_cost(trk, det)
                if math.isfinite(cost):
                    pairs.append((cost, tid, did))

        # 使用贪心匹配：
        # 对中小规模目标数，足够简单，也比较快。
        # 如果后续目标很多，可以换成匈牙利算法。
        pairs.sort(key=lambda x: x[0])
        used_t = set()
        used_d = set()
        for _, tid, did in pairs:
            if tid in used_t or did in used_d:
                continue
            self.tracks[tid].update(detections[did])
            used_t.add(tid)
            used_d.add(did)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(did)

        # 没匹配上的旧轨迹记为 miss
        for tid in unmatched_tracks:
            self.tracks[tid].miss()

        # 没匹配上的新检测创建新轨迹
        for did in unmatched_dets:
            det = detections[did]
            smooth = SmoothedClassState(alpha=self.cfg.class_smooth_alpha)
            smooth.update(det.class_id, det.conf)
            trk = Track(
                track_id=self.next_id,
                kf=Kalman2D(det.cx, det.cy),
                smooth_class=smooth,
                last_box=det.xyxy.copy(),
                last_conf=det.conf,
                history=deque(maxlen=self.cfg.track_history),
            )
            trk.history.append((int(det.cx), int(det.cy)))
            self.tracks[self.next_id] = trk
            self.next_id += 1

        # 删除长时间丢失的轨迹，防止轨迹表无限增长
        stale = [tid for tid, trk in self.tracks.items() if trk.lost_frames > self.cfg.max_lost_frames]
        for tid in stale:
            self.tracks.pop(tid, None)

        return list(self.tracks.values())


# =============================
# 6) 目标选择器（按 track 决策）
# =============================
class TargetSelector:
    """
    从多个轨迹中选出“当前要跟随的那个目标”。

    为什么一定要按 track 选，不按 box 选：
    - box 是单帧结果，天然容易跳
    - track 有历史，有类别平滑，有连续性，适合做稳定决策
    """
    def __init__(self, cfg: AppConfig, frame_center: Tuple[float, float]):
        self.cfg = cfg
        self.frame_center = frame_center
        self.current_target_id: Optional[int] = None
        self.last_seen_target_ts: float = 0.0

    def _class_allowed(self, class_name: str) -> bool:
        """判断该类别是否允许被跟随。"""
        if self.cfg.class_allowlist and class_name not in self.cfg.class_allowlist:
            return False
        if class_name in self.cfg.class_blocklist:
            return False
        return True

    def select(self, tracks: List[Track], class_names: Tuple[str, ...]) -> Optional[Track]:
        """
        目标选择逻辑。

        打分依据：
        - 越靠近画面中心越优先
        - 检测置信度越高越优先
        - 类别平滑后越稳定越优先
        - 当前已锁定目标会获得额外粘性分
        - 丢失帧越多，分数越低
        """
        cx0, cy0 = self.frame_center
        candidates: List[Tuple[float, Track]] = []

        for trk in tracks:
            # 未确认的新轨迹先不参与决策，减少误检切入
            if not trk.is_confirmed(self.cfg.min_confirmed_hits):
                continue

            class_id = trk.stable_class_id
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"cls_{class_id}"
            if not self._class_allowed(class_name):
                continue

            px, py = trk.center
            dist = math.hypot(px - cx0, py - cy0)

            # 这里给一个简单可解释的打分公式。
            # 中心距离是主要项；检测置信度和类别稳定性是辅助项。
            score = -dist + 25.0 * trk.last_conf + 15.0 * trk.smooth_class.confidence()

            # 对当前已锁定目标增加粘性，减少多目标来回切换
            if self.current_target_id == trk.track_id:
                score += self.cfg.target_stickiness_bonus

            # 丢失帧越多，说明状态越不可信
            score -= trk.lost_frames * 18.0
            candidates.append((score, trk))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0][1]
            self.current_target_id = chosen.track_id
            self.last_seen_target_ts = now()
            return chosen

        # 如果没有新候选，但当前目标只是短暂消失，则给一点宽限期
        if self.current_target_id is not None and (now() - self.last_seen_target_ts) < self.cfg.lost_target_grace_s:
            for trk in tracks:
                if trk.track_id == self.current_target_id:
                    return trk

        self.current_target_id = None
        return None


# =============================
# 7) 视角控制器（PD）
# =============================
class PTZCommand:
    """
    发送给外部设备的一次控制命令。

    dx / dy 表示每个控制周期内的水平/垂直增量。
    ts 用来做超时保护。
    """
    __slots__ = ("dx", "dy", "ts")

    def __init__(self, dx: float, dy: float, ts: float):
        self.dx = dx
        self.dy = dy
        self.ts = ts


class CameraPTZController:
    """
    把“目标相对于画面中心的偏差”转换成“相机/云台控制量”。

    为什么使用 PD：
    - P 项：误差越大，输出越大
    - D 项：根据误差变化速度抑制过冲
    - 不默认启用 I 项，是为了减少视觉系统中的积分累积问题
    """
    def __init__(self, cfg: AppConfig, frame_center: Tuple[float, float]):
        self.cfg = cfg
        self.frame_center = frame_center
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.last_ts = now()

    def compute(self, target_center: Optional[Tuple[float, float]]) -> PTZCommand:
        """
        根据当前目标中心，计算一条平滑控制命令。

        为什么这里一定要用 dt：
        - 30 FPS 和 50 FPS 下，相邻两帧时间差不同
        - D 项如果不除以 dt，会因为帧率变化而变形
        """
        ts = now()
        dt = max(1 / 240.0, ts - self.last_ts)
        self.last_ts = ts

        # 没有目标时，输出归零，同时清空历史误差，避免目标重新出现时突然冲一下
        if target_center is None:
            self.prev_error_x = 0.0
            self.prev_error_y = 0.0
            return PTZCommand(0.0, 0.0, ts)

        cx0, cy0 = self.frame_center
        tx, ty = target_center
        err_x = tx - cx0
        err_y = ty - cy0

        # 死区：接近中心时不再微调，减少抖动
        if abs(err_x) < self.cfg.deadzone_px:
            err_x = 0.0
        if abs(err_y) < self.cfg.deadzone_px:
            err_y = 0.0

        derr_x = (err_x - self.prev_error_x) / dt
        derr_y = (err_y - self.prev_error_y) / dt
        self.prev_error_x = err_x
        self.prev_error_y = err_y

        out_x = self.cfg.kp_x * err_x + self.cfg.kd_x * derr_x
        out_y = self.cfg.kp_y * err_y + self.cfg.kd_y * derr_y

        # 输出限幅：防止突发误差或错误切换引起的大跳变
        out_x = clamp(out_x, -self.cfg.max_delta_per_tick, self.cfg.max_delta_per_tick)
        out_y = clamp(out_y, -self.cfg.max_delta_per_tick, self.cfg.max_delta_per_tick)
        return PTZCommand(out_x, out_y, ts)


# =============================
# 8) 串口输出线程
# =============================
class SerialPTZWriter:
    """
    串口发送线程。

    协议：<hh>
    - 发送两个 int16: dx, dy

    为什么单独开线程而不是图像线程里直接 write：
    - 串口写入可能偶尔阻塞
    - 图像线程的目标是尽量稳定抓图、推理、更新轨迹
    - 把 I/O 放到单独线程里，整体抖动会更小

    为什么使用队列：
    - 避免多个线程直接共享一个“全局 dx/dy”，那样容易有竞态
    - 队列天然适合做“最新命令覆盖旧命令”的逻辑
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.q: "queue.Queue[PTZCommand]" = queue.Queue(maxsize=4)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.ser: Optional[serial.Serial] = None

    def start(self):
        """打开串口并启动发送线程。"""
        if not self.cfg.serial_enabled:
            return
        self.ser = serial.Serial(self.cfg.com_port, self.cfg.baud_rate, write_timeout=0)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止发送线程并关闭串口。"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def submit(self, cmd: PTZCommand):
        """
        提交一条新命令。

        为什么队列满了要丢旧命令：
        - 控制系统最重要的是“当前最新状态”
        - 老命令如果积压，会导致设备做过时动作
        """
        if not self.cfg.serial_enabled:
            return
        while self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        try:
            self.q.put_nowait(cmd)
        except queue.Full:
            pass

    def _loop(self):
        """
        固定频率发送最新命令。

        为什么采用固定 control_hz：
        - 外设控制通常希望更平滑、稳定的刷新节奏
        - 不应该完全受检测帧率波动影响
        """
        period = 1.0 / self.cfg.control_hz
        last_cmd = PTZCommand(0.0, 0.0, now())
        while self.running:
            t0 = now()
            try:
                # 每个周期只保留最新命令
                while True:
                    last_cmd = self.q.get_nowait()
            except queue.Empty:
                pass

            # 超时保护：如果图像侧太久没给新命令，则自动停转
            if now() - last_cmd.ts > self.cfg.command_timeout_s:
                send_dx, send_dy = 0, 0
            else:
                send_dx = int(clamp(round(last_cmd.dx), -32767, 32767))
                send_dy = int(clamp(round(last_cmd.dy), -32767, 32767))

            if self.ser is not None:
                try:
                    packet = np.array([send_dx, send_dy], dtype=np.int16).tobytes()
                    self.ser.write(packet)
                except serial.SerialException:
                    # 串口偶发异常时，这里先忽略，避免主流程崩溃
                    pass

            sleep_s = period - (now() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)


# =============================
# 9) Windows 热键读取
# =============================
def vk_pressed(vk_code: int) -> bool:
    """
    检查某个虚拟键是否按下。

    这里使用 Windows 的 GetAsyncKeyState，适合做简单热键控制。
    """
    import ctypes
    return (ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


# =============================
# 10) 调试可视化绘制
# =============================
def draw_overlay(frame: np.ndarray,
                 tracks: List[Track],
                 target: Optional[Track],
                 class_names: Tuple[str, ...],
                 fps: float,
                 center_xy: Tuple[int, int]):
    """
    在窗口上绘制轨迹、类别、目标和 FPS。

    为什么建议保留这个函数：
    - 实时系统里“看见内部状态”非常重要
    - 很多问题不是看日志能看出来的，而是轨迹一画就知道哪里不稳
    """
    cx0, cy0 = center_xy
    cv2.line(frame, (cx0 - 12, cy0), (cx0 + 12, cy0), (0, 255, 255), 2)
    cv2.line(frame, (cx0, cy0 - 12), (cx0, cy0 + 12), (0, 255, 255), 2)

    for trk in tracks:
        x1, y1, x2, y2 = map(int, trk.last_box)
        px, py = map(int, trk.center)
        class_id = trk.stable_class_id
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"cls_{class_id}"
        is_target = target is not None and trk.track_id == target.track_id
        color = (0, 255, 0) if is_target else (255, 160, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (px, py), 3, color, -1)
        label = f"ID {trk.track_id} | {class_name} | c={trk.last_conf:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 绘制历史轨迹，让你能直观看到轨迹是否平滑、是否频繁跳变
        hist = list(trk.history)
        for i in range(1, len(hist)):
            cv2.line(frame, hist[i - 1], hist[i], color, 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


# =============================
# 11) 主应用流程
# =============================
class TrackingApp:
    """
    把所有模块组装起来：
    抓图 -> 检测 -> 跟踪 -> 目标选择 -> 控制 -> 串口输出 -> 可视化

    为什么主流程要尽量清晰：
    - 课程展示时，这种“单向数据流”非常容易讲清楚
    - 后续优化性能时，也方便定位哪一段最耗时
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.region = cfg.region
        self.region_w = self.region[2] - self.region[0]
        self.region_h = self.region[3] - self.region[1]
        self.center = (self.region_w // 2, self.region_h // 2)

        self.detector = DetectorTRT(cfg)
        self.tracker = MultiObjectTracker(cfg)
        self.selector = TargetSelector(cfg, self.center)
        self.controller = CameraPTZController(cfg, self.center)
        self.writer = SerialPTZWriter(cfg)

        # bettercam 用于高频截图
        self.camera = bettercam.create(output_idx=cfg.output_idx, output_color="BGR")
        self.fps_timer = now()
        self.frame_counter = 0
        self.fps = 0.0
        self.show_window = cfg.show_window
        self.last_f9_state = False

    def run(self):
        """
        主循环。

        每帧流程：
        1) 抓图
        2) 检测
        3) 跟踪更新
        4) 选择当前目标
        5) 计算控制命令
        6) 提交给串口线程
        7) 更新可视化窗口
        """
        print("Starting safe camera tracker...")
        if self.cfg.serial_enabled:
            print(f"Opening PTZ serial link on {self.cfg.com_port} @ {self.cfg.baud_rate}...")
            self.writer.start()

        try:
            while True:
                # END 退出
                if vk_pressed(self.cfg.vk_end):
                    print("END pressed. Exiting.")
                    break

                # F9 显示/隐藏调试窗口
                # 为什么提供这个开关：
                # - 渲染窗口也会吃性能
                # - 演示和调试时打开，正式跑性能时可以关掉
                f9 = vk_pressed(self.cfg.vk_f9)
                if f9 and not self.last_f9_state:
                    self.show_window = not self.show_window
                    if not self.show_window:
                        cv2.destroyAllWindows()
                self.last_f9_state = f9

                frame = self.camera.grab(region=self.region)
                if frame is None:
                    time.sleep(0.002)
                    continue

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections)
                target = self.selector.select(tracks, self.cfg.class_names)
                target_center = target.center if target is not None else None
                cmd = self.controller.compute(target_center)
                self.writer.submit(cmd)

                # FPS 统计
                self.frame_counter += 1
                elapsed = now() - self.fps_timer
                if elapsed >= 1.0:
                    self.fps = self.frame_counter / elapsed
                    self.frame_counter = 0
                    self.fps_timer = now()

                if self.show_window:
                    vis = frame.copy()
                    draw_overlay(vis, tracks, target, self.cfg.class_names, self.fps, self.center)
                    cv2.imshow("Safe Camera Tracker (PTZ)", vis)
                    cv2.waitKey(1)
        finally:
            self.writer.stop()
            cv2.destroyAllWindows()


# =============================
# 12) 程序入口
# =============================
if __name__ == "__main__":
    # 这里给的是示例配置。
    # 你需要根据自己的环境修改：
    # - engine_path
    # - 截图区域 region
    # - 串口号 com_port
    # - 类别名 class_names
    # - 允许跟随的类别 class_allowlist
    cfg = AppConfig(
        engine_path=r"D:/models/your_model.engine",
        region=(768, 208, 1792, 1232),
        imgsz=896,
        show_window=True,
        serial_enabled=True,
        com_port="COM11",
        baud_rate=921600,
        class_names=("person", "friend", "other"),
        class_allowlist=("person",),
    )

    # 创建并运行主应用
    app = TrackingApp(cfg)
    app.run()
