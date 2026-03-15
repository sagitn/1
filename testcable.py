import cv2
import numpy as np
import time
import ctypes
import struct
import math
import bettercam  # 替换为 BetterCam
import serial
import threading
import tensorrt as trt
import torch
import torchvision

# ================= 1. 串口硬件配置 =================
COM_PORT = "COM11"  
BAUD_RATE = 921600  

print(f"🔌 正在尝试连接硬件串口 {COM_PORT} ...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, write_timeout=0) 
    print("✅ 硬件串口连接成功！")
except Exception as e:
    print(f"❌ 串口连接失败: {e}\n请检查端口号是否填写正确，或者 Arduino 串口监视器是否占用了该端口。")
    exit()

# ================= 2. 触发与模式配置 =================
USE_TOGGLE_MODE = True
VK_F9 = 0x78    
VK_END = 0x23   
# 【关键修改】默认关闭绘图和窗口，极大提升 FPS
show_window = False 

# ================= 3. 吸附 (Aimbot) 配置 =================
AIM_SMOOTH = 0.3    
AIM_OFFSET_Y = -5   
X_SENSITIVITY_RATIO = 0.5  
Y_SENSITIVITY_RATIO = 0.5
AIM_DEADZONE = 7          
MAX_STEP = 80             

# --- 新增：基于时间的动态加速配置 ---
LAG_DELAY = 0.20       # 延迟时间：300ms (0.3秒) 内没追上才开始加速
LAG_MAX_MULT = 3.0     # 最大加速倍数：最多放大到原来速度的 3 倍
LAG_RAMP_UP = 10.0      # 加速斜率：超过 300ms 后，拉力倍数每秒增加多少 (5.0 意味着第0.5秒时倍数达到 2.0 倍)
LAG_TOLERANCE = 10     # 容忍距离(像素)：误差大于这个值才算"跟不上"，防止微调时误触发加速

# ================= 4. 压枪配置 =================
DOWN_INTERVAL = 0.016  
RECOIL_PATTERN = [
    [0, 4], [0, 3], [0, 3], [0, 4], [0, 4], [0, 5], [0, 6], [0, 6],
    # [0, 7], [0, 9], [0, 7], [0, 7], [0, 8], [0, 7], [0, 9], [0, 8],
    # [0, 7], [0, 9], [0, 8], [0, 11], [0, 9], [0, 11], [0, 11], [0, 11],
    # [0, 7], [0, 9], [0, 8], [0, 11], [0, 9], [0, 11], [0, 11], [0, 11],
    # [0, 7], [0, 9], [0, 8], [0, 11], [0, 9], [0, 11], [0, 11], [0, 11],
]

# ================= 5. 多线程共享全局变量 =================
shared_aim_dx = 0.0
shared_aim_dy = 0.0
global_is_firing = False
global_firing_start_time = 0.0
program_running = True 

# ================= 6. 物理控制子线程 (1000Hz) =================
def hardware_control_loop():
    global shared_aim_dx, shared_aim_dy, program_running
    global global_is_firing, global_firing_start_time

    caps_physical_was_pressed = False
    caps_active = (ctypes.windll.user32.GetKeyState(0x14) & 0x0001) != 0 
    last_caps_state = caps_active  
    
    is_firing = False
    recoil_index = 0
    last_recoil_time = 0.0
    
    while program_running:
        loop_start = time.perf_counter()
        
        if USE_TOGGLE_MODE:
            caps_physical_pressed = (ctypes.windll.user32.GetAsyncKeyState(0x14) & 0x8000) != 0
            if caps_physical_pressed and not caps_physical_was_pressed:
                caps_active = not caps_active  
            caps_physical_was_pressed = caps_physical_pressed
            lmb_pressed = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
            trigger_aim = caps_active
            trigger_recoil = caps_active and lmb_pressed
        else:
            caps_active = (ctypes.windll.user32.GetAsyncKeyState(0x14) & 0x8000) != 0
            lmb_pressed = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
            trigger_aim = caps_active
            trigger_recoil = caps_active and lmb_pressed

        if caps_active != last_caps_state:
            if caps_active:
                print("🔄 大写锁定触发 -> AI 助手状态: 🟢 开启 (ARMED)")
            else:
                print("🔄 大写锁定触发 -> AI 助手状态: 🔴 关闭 (DISARMED)")
            last_caps_state = caps_active

        final_dx = 0.0
        final_dy = 0.0
        current_time = time.perf_counter()

        if trigger_recoil:
            if not is_firing:
                is_firing = True
                recoil_index = 0
                last_recoil_time = current_time
                global_is_firing = True
                global_firing_start_time = current_time
            
            if current_time - last_recoil_time >= DOWN_INTERVAL:
                if recoil_index < len(RECOIL_PATTERN):
                    rdx, rdy = RECOIL_PATTERN[recoil_index]
                    recoil_index += 1
                else:
                    rdx, rdy = RECOIL_PATTERN[-1]
                
                final_dx += rdx
                final_dy += rdy
                last_recoil_time = current_time
        else:
            if is_firing:
                is_firing = False
                global_is_firing = False 

        if trigger_aim:
            final_dx += shared_aim_dx
            final_dy += shared_aim_dy
            shared_aim_dx = 0.0
            shared_aim_dy = 0.0

        if final_dx != 0 or final_dy != 0:
            send_x = max(-127, min(127, int(final_dx)))
            send_y = max(-127, min(127, int(final_dy)))
            data = struct.pack('<hh', send_x, send_y)
            try:
                ser.write(data)
            except:
                pass

        while time.perf_counter() - loop_start < 0.001:
            time.sleep(0)

control_thread = threading.Thread(target=hardware_control_loop, daemon=True)
control_thread.start()
print("⚡ 独立硬件控制线程已启动 (轮询率: 1000Hz)")

# ================= 7. 原生 TensorRT 10 引擎封装 =================
class TRTEngine:
    def __init__(self, engine_path, device=torch.device('cuda:0')):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.stream = torch.cuda.Stream(device=device)

        # 针对 TensorRT 10.x 的新版内存绑定逻辑
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            # 处理动态 batch size
            if shape[0] == -1: 
                shape = (1,) + shape[1:]
                self.context.set_input_shape(name, shape)

            # 在显存中分配 Torch 张量
            tensor = torch.empty(tuple(shape), dtype=torch.from_numpy(np.empty(0, dtype=dtype)).dtype, device=device)
            
            # TRT 10 核心：直接通过名称映射显存指针
            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if is_input:
                self.inputs.append({'name': name, 'tensor': tensor, 'shape': shape})
            else:
                self.outputs.append({'name': name, 'tensor': tensor, 'shape': shape})

    def infer(self, img_tensor):
        # 拷贝图像数据到输入张量
        self.inputs[0]['tensor'].copy_(img_tensor)
        
        # TRT 10 核心：使用 execute_async_v3 替代废弃的 v2
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        # 同步流，等待计算完成
        self.stream.synchronize()
        return self.outputs[0]['tensor']

# ================= 8. 主程序初始化 =================
print("🚀 正在加载原生 TensorRT 引擎 (跳过 Ultralytics)...")
trt_model = TRTEngine('D:/S/Documents/Syncthing/PUBG-AI/PUBG_YOLO/m24best_pure.engine')
# trt_model = TRTEngine('D:/S/Documents/Syncthing/PUBG-AI/PUBG_YOLO/probest_pure.engine')

REGION = (768, 208, 1792, 1232)
REGION_WIDTH = REGION[2] - REGION[0]
REGION_HEIGHT = REGION[3] - REGION[1]
CENTER_X, CENTER_Y = 512, 512

# 模型参数
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMGSZ = 1024
RATIO_X = REGION_WIDTH / IMGSZ
RATIO_Y = REGION_HEIGHT / IMGSZ

print("📸 正在初始化 BetterCam (超低延迟模式)...")
camera = bettercam.create(output_idx=0, output_color="BGR")

print("\n✅ AI 视觉主线程已就绪！(默认后台静默运行，按 F9 显示画面)")

last_f9_state = False
ai_fps_start_time = time.perf_counter()
ai_frame_count = 0
current_ai_fps = 0.0

try:
    # === 追踪记忆变量 ===
    locked_target_pos = None  # 记录上一帧锁定的目标坐标 (x, y)
    LOCK_THRESHOLD = 150      # 追踪阈值(像素)：如果目标一帧内移动超过这个距离，视为跟丢，重新寻找新目标

    # --- 新增：超时加速计时器 ---
    tracking_lag_start = None

    while True:
        if (ctypes.windll.user32.GetAsyncKeyState(VK_END) & 0x8000) != 0:
            print("\n🛑 收到 END 键指令，正在安全关闭 AI 引擎...")
            program_running = False
            break

        # F9 切换显示状态
        current_f9_state = (ctypes.windll.user32.GetAsyncKeyState(VK_F9) & 0x8000) != 0
        if current_f9_state and not last_f9_state:
            show_window = not show_window
            if not show_window:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                print("🙈 识别窗口已隐藏，开启极限性能模式。")
            else:
                print("👀 识别窗口已恢复显示。")
        last_f9_state = current_f9_state

        # BetterCam 极速抓取
        frame = camera.grab(region=REGION)
        if frame is None:
            time.sleep(0.002)
            continue
        
        # === 纯 GPU 预处理 ===
        img = cv2.resize(frame, (IMGSZ, IMGSZ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).cuda().float().unsqueeze(0) / 255.0

        # === 原生 TRT 推理 ===
        # YOLOv11 输出形状通常为 [1, 5, 8400] (假设单类别)
        preds = trt_model.infer(img_tensor).squeeze(0).transpose(0, 1) # 变成 [8400, 5]
        
        # === GPU 内非极大值抑制 (NMS) ===
        scores = preds[:, 4]
        mask = scores > CONF_THRESHOLD
        valid_preds = preds[mask]
        
        temp_aim_dx = 0.0
        temp_aim_dy = 0.0
        final_boxes = []

        current_time_vis = time.perf_counter()
        dynamic_center_y = CENTER_Y  
        firing_duration = 0.0
        
        if global_is_firing:
            firing_duration = current_time_vis - global_firing_start_time
            y_shift = np.interp(firing_duration, [0, 0.1, 0.2, 0.4, 0.6], [0, 5, 15, 25, 30])
            dynamic_center_y = CENTER_Y - y_shift

        if len(valid_preds) > 0:
            boxes = valid_preds[:, :4]
            # cx, cy, w, h 转 x1, y1, x2, y2
            boxes_xyxy = torch.empty_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            
            keep = torchvision.ops.nms(boxes_xyxy, valid_preds[:, 4], IOU_THRESHOLD)
            final_targets = valid_preds[keep].cpu().numpy()

            min_dist = float('inf')
            best_target = None
            
            ##############################################
            # for target in final_targets:
            #     # 还原到 1024 截图的真实像素坐标
            #     cx = target[0] * RATIO_X
            #     cy = target[1] * RATIO_Y
            #     w = target[2] * RATIO_X
            #     h = target[3] * RATIO_Y
            #     final_boxes.append((cx, cy, w, h))

            #     dist = math.hypot(cx - CENTER_X, cy - CENTER_Y)
            #     if dist < min_dist:
            #         min_dist = dist
            #         best_target = (cx, cy)
            ###############################################
            
            # 1. 先把所有框的实际坐标算出来，存入列表
            current_frame_targets = []
            for target in final_targets:
                cx = target[0] * RATIO_X
                cy = target[1] * RATIO_Y
                w = target[2] * RATIO_X
                h = target[3] * RATIO_Y
                final_boxes.append((cx, cy, w, h))
                current_frame_targets.append((cx, cy))

            # 2. 如果当前有记忆的锁定目标，优先在当前帧寻找 "同一个他"
            if locked_target_pos is not None:
                min_track_dist = float('inf')
                for cx, cy in current_frame_targets:
                    # 计算与【上一帧目标位置】的距离，而不是与【屏幕中心】的距离
                    dist_to_lock = math.hypot(cx - locked_target_pos[0], cy - locked_target_pos[1])
                    if dist_to_lock < min_track_dist:
                        min_track_dist = dist_to_lock
                        best_target = (cx, cy)
                
                # 如果找到的最接近目标跨度太大（比如敌人死了或者瞬移），说明跟丢了，清除锁定
                if min_track_dist > LOCK_THRESHOLD:
                    locked_target_pos = None
                    best_target = None

            # 3. 如果没有锁定目标（或者刚刚跟丢了），则回退到基础逻辑：寻找离准星最近的新目标
            if locked_target_pos is None:
                min_dist = float('inf')
                for cx, cy in current_frame_targets:
                    dist = math.hypot(cx - CENTER_X, cy - CENTER_Y)
                    if dist < min_dist:
                        min_dist = dist
                        best_target = (cx, cy)

            # 4. 更新记忆变量，留给下一帧使用
            if best_target:
                locked_target_pos = best_target
            else:
                # 如果当前帧画面里没有任何敌人，清空锁定记忆和加速计时器
                locked_target_pos = None
                temp_aim_dx = 0.0
                temp_aim_dy = 0.0
                tracking_lag_start = None  # 👈 重要：丢失目标必须重置计时器

            # ============= 核心计算：引入时间动态增益 =============
            if best_target:
                ex, ey = best_target
                pixel_error_x = ex - CENTER_X
                pixel_error_y = ey - dynamic_center_y - AIM_OFFSET_Y
                
                # 计算当前总误差距离 (直线距离)
                error_dist = math.hypot(pixel_error_x, pixel_error_y)
                
                # --- 基于时间的动态加速逻辑 ---
                dynamic_multiplier = 1.0  # 默认不加速
                
                if error_dist > LAG_TOLERANCE:
                    # 1. 误差较大（大于容忍值），开始计时或计算持续时间
                    if tracking_lag_start is None:
                        tracking_lag_start = time.perf_counter()
                    else:
                        lag_duration = time.perf_counter() - tracking_lag_start
                        
                        # 2. 如果持续"跟不上"的时间超过了 300ms
                        if lag_duration > LAG_DELAY:
                            extra_time = lag_duration - LAG_DELAY
                            # 随着时间的推移，线性增加加速倍数
                            dynamic_multiplier += extra_time * LAG_RAMP_UP
                            # 限制最高只能加速到 LAG_MAX_MULT 倍
                            dynamic_multiplier = min(dynamic_multiplier, LAG_MAX_MULT)
                else:
                    # 3. 追上了（误差小于容忍值），立刻清空计时器，恢复普通平滑拉枪
                    tracking_lag_start = None
                
                # 死区处理 (在容忍值以内的微调还是受死区控制)
                if abs(pixel_error_x) <= AIM_DEADZONE: pixel_error_x = 0
                if abs(pixel_error_y) <= AIM_DEADZONE: pixel_error_y = 0

                # 1. 计算基础线性速度，并【乘上动态加速倍率】
                temp_aim_dx = pixel_error_x * X_SENSITIVITY_RATIO * AIM_SMOOTH * 0.8 * dynamic_multiplier
                temp_aim_dy = pixel_error_y * Y_SENSITIVITY_RATIO * AIM_SMOOTH * 0.5 * dynamic_multiplier

                # 2. 保底起步速度（解决微小距离死区外粘滞）
                MIN_MOVE = 5.5  
                if temp_aim_dx != 0:
                    temp_aim_dx = temp_aim_dx + math.copysign(MIN_MOVE, temp_aim_dx)
                if temp_aim_dy != 0:
                    temp_aim_dy = temp_aim_dy + math.copysign(MIN_MOVE, temp_aim_dy)
                
                # 3. 限制单次移动的最大步幅（防瞎甩）
                temp_aim_dx = max(-MAX_STEP, min(MAX_STEP, temp_aim_dx))
                temp_aim_dy = max(-MAX_STEP, min(MAX_STEP, temp_aim_dy))

        shared_aim_dx = temp_aim_dx
        shared_aim_dy = temp_aim_dy

        ai_frame_count += 1
        if time.perf_counter() - ai_fps_start_time >= 1.0:
            current_ai_fps = ai_frame_count / (time.perf_counter() - ai_fps_start_time)
            ai_frame_count = 0
            ai_fps_start_time = time.perf_counter()

        # === 画面渲染 (只有 F9 开启时才执行，节省海量算力) ===
        if show_window:
            annotated_frame = frame.copy()
            # 绘制准心
            cv2.line(annotated_frame, (CENTER_X - 10, CENTER_Y), (CENTER_X + 10, CENTER_Y), (0, 255, 0), 2)
            cv2.line(annotated_frame, (CENTER_X, CENTER_Y - 10), (CENTER_X, CENTER_Y + 10), (0, 255, 0), 2)
            
            dynamic_y_int = int(dynamic_center_y)
            cv2.line(annotated_frame, (CENTER_X - 10, dynamic_y_int), (CENTER_X + 10, dynamic_y_int), (0, 0, 255), 2)
            cv2.line(annotated_frame, (CENTER_X, dynamic_y_int - 10), (CENTER_X, dynamic_y_int + 10), (0, 0, 255), 2)

            # 手动绘制识别框
            for (cx, cy, w, h) in final_boxes:
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"AI FPS: {current_ai_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if global_is_firing:
                debug_info = f"Fire Time: {firing_duration:.2f}s | Center Y: {dynamic_y_int}"
                cv2.putText(annotated_frame, debug_info, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            cv2.imshow("PUBG - AI Engine (Native TRT)", annotated_frame)
            cv2.waitKey(1)

except KeyboardInterrupt:
    print("\n程序已被强行终止。")
    program_running = False
finally:
    control_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
    if 'ser' in locals() and ser.is_open:
        ser.close()
    print("✅ 所有线程和硬件通讯端口已安全释放。")