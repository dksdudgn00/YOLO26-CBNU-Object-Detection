#!/usr/bin/env python3
"""
CARLA 0.9.14 + YOLO26 실시간 추론
────────────────────────────────────────────────────────
1. CARLA 서버에 접속
2. 차량 스폰 + RGB 카메라 장착
3. 카메라 프레임마다 YOLO26 추론
4. OpenCV 창으로 실시간 시각화
5. 결과 영상 mp4 저장 (선택)

실행 전 CARLA 서버 먼저 시작:
  cd /home/a/Desktop/yhan/Sources/CARLA_0.9.14
  ./CarlaUE4.sh -RenderOffScreen   # 헤드리스 모드 (GUI 없이)
  ./CarlaUE4.sh                    # GUI 모드
"""

import sys
import time
import queue
import random
import threading
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# CARLA Python API 경로 추가
CARLA_PATH = Path("/home/a/Desktop/yhan/Sources/CARLA_0.9.14")
sys.path.append(str(CARLA_PATH / "PythonAPI/carla/dist/carla-0.9.14-py3.10-linux-x86_64.egg"))
sys.path.append(str(CARLA_PATH / "PythonAPI/carla"))
import carla

# ── Configuration ─────────────────────────────────────────────────────────────
WORK_DIR      = Path("/home/a/Desktop/yhan/Sources/custom_yolo_train")
WEIGHTS       = WORK_DIR / "runs/yolo26_train/weights/best.pt"  # fine-tuned
# WEIGHTS     = WORK_DIR / "yolo26n.pt"                         # pre-trained
CARLA_HOST    = "localhost"
CARLA_PORT    = 2000
CAM_WIDTH     = 1280
CAM_HEIGHT    = 720
CAM_FOV       = 90
CONF          = 0.35
IOU           = 0.45
SAVE_VIDEO    = True
AUTOPILOT     = True   # 자동 주행 여부
SPAWN_NPC     = True   # NPC 차량/보행자 스폰 여부
N_NPC_CARS    = 30
N_NPC_PEDS    = 20
# ──────────────────────────────────────────────────────────────────────────────

CLASSES    = ["person", "tree", "vehicle"]
COLORS     = {0: (0, 200, 50), 1: (34, 139, 34), 2: (30, 120, 255)}


def draw_box(img, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(img, frame_idx, fps, counts):
    lines = [
        f"Frame: {frame_idx}",
        f"FPS:   {fps:.1f}",
        f"Model: {Path(WEIGHTS).stem}",
    ] + [f"{CLASSES[c]}: {n}" for c, n in sorted(counts.items())]
    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


def spawn_npcs(world, traffic_manager, n_cars, n_peds):
    """NPC 차량과 보행자 스폰"""
    spawned = {"cars": [], "peds": [], "ped_controllers": []}
    bp_lib = world.get_blueprint_library()

    # 차량
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for i, sp in enumerate(spawn_points[:n_cars]):
        bp = random.choice(vehicle_bps)
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, traffic_manager.get_port())
            spawned["cars"].append(actor)

    # 보행자 — 2단계 스폰 (서버 tick 동기화 필수)
    ped_bps = bp_lib.filter("walker.pedestrian.*")
    ctrl_bp = bp_lib.find("controller.ai.walker")

    # 1단계: 보행자 actor 먼저 전부 스폰
    ped_actors = []
    for _ in range(n_peds):
        sp = world.get_random_location_from_navigation()
        if sp is None:
            continue
        bp = random.choice(ped_bps)
        ped = world.try_spawn_actor(bp, carla.Transform(sp))
        if ped:
            ped_actors.append(ped)

    # 서버가 actor 등록 완료할 때까지 대기
    world.wait_for_tick()

    # 2단계: 컨트롤러 부착 및 시작
    for ped in ped_actors:
        try:
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=ped)
            world.wait_for_tick()
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.4)
            spawned["peds"].append(ped)
            spawned["ped_controllers"].append(ctrl)
        except Exception as e:
            print(f"  보행자 컨트롤러 실패 (무시): {e}")
            try:
                ped.destroy()
            except Exception:
                pass

    print(f"  NPC: 차량 {len(spawned['cars'])}대  보행자 {len(spawned['peds'])}명")
    return spawned


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  CARLA + YOLO26 실시간 추론")
    print(f"  모델  : {WEIGHTS.name}")
    print(f"  서버  : {CARLA_HOST}:{CARLA_PORT}")
    print(f"  카메라: {CAM_WIDTH}x{CAM_HEIGHT}")
    print(f"  장치  : {'GPU' if device == 0 else 'CPU'}")
    print(f"{'='*55}\n")

    model = YOLO(str(WEIGHTS))
    print("  YOLO 모델 로드 완료")

    # CARLA 연결
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()
    tm     = client.get_trafficmanager(8000)
    print(f"  CARLA 서버 연결 완료  (맵: {world.get_map().name})")

    # 날씨 설정 (선택)
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)

    actor_list = []
    npcs = {"cars": [], "peds": [], "ped_controllers": []}

    try:
        bp_lib = world.get_blueprint_library()

        # ── 차량 스폰 ──────────────────────────────────────────────────────
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_pts  = world.get_map().get_spawn_points()
        spawn_tf   = random.choice(spawn_pts)
        vehicle    = world.spawn_actor(vehicle_bp, spawn_tf)
        actor_list.append(vehicle)
        print(f"  차량 스폰: {vehicle.type_id}  @ {spawn_tf.location}")

        if AUTOPILOT:
            vehicle.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(vehicle, 0)

        # ── RGB 카메라 장착 ────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        cam_bp.set_attribute("fov",          str(CAM_FOV))
        cam_transform = carla.Transform(
            carla.Location(x=2.0, z=1.5),   # 차량 전방 상단
            carla.Rotation(pitch=-10)
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)
        world.wait_for_tick()   # 카메라 등록 대기
        print("  RGB 카메라 장착 완료")

        # ── NPC 스폰 ───────────────────────────────────────────────────────
        if SPAWN_NPC:
            npcs = spawn_npcs(world, tm, N_NPC_CARS, N_NPC_PEDS)

        # ── 카메라 프레임 큐 ───────────────────────────────────────────────
        frame_queue = queue.Queue(maxsize=4)

        def on_image(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))   # BGRA
            bgr = arr[:, :, :3].copy()
            if not frame_queue.full():
                frame_queue.put(bgr)

        camera.listen(on_image)

        # ── VideoWriter ────────────────────────────────────────────────────
        writer = None
        if SAVE_VIDEO:
            out_path = WORK_DIR / "carla_inference.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 20.0,
                                     (CAM_WIDTH, CAM_HEIGHT))
            print(f"  영상 저장: {out_path}")

        print("\n  추론 시작  (q / ESC 로 종료)\n")

        frame_idx = 0
        fps       = 0.0
        t_prev    = time.time()

        while True:
            try:
                frame = frame_queue.get(timeout=2.0)
            except queue.Empty:
                print("  카메라 프레임 수신 대기 중...")
                continue

            frame_idx += 1

            # ── YOLO 추론 ──────────────────────────────────────────────────
            t0      = time.time()
            results = model(frame, conf=CONF, iou=IOU,
                            device=device, verbose=False)[0]
            fps     = 1.0 / max(time.time() - t0, 1e-6)

            counts = {}
            if results.boxes is not None:
                for box in results.boxes:
                    cid  = int(box.cls[0])
                    cf   = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = CLASSES[cid] if cid < len(CLASSES) else str(cid)
                    color = COLORS.get(cid, (200, 200, 200))
                    draw_box(frame, x1, y1, x2, y2, label, cf, color)
                    counts[cid] = counts.get(cid, 0) + 1

            draw_hud(frame, frame_idx, fps, counts)

            # ── 표시 & 저장 ────────────────────────────────────────────────
            cv2.imshow("CARLA + YOLO26", frame)
            if writer:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q / ESC
                print("\n  사용자 종료")
                break

    finally:
        # 정리
        if 'writer' in dir() and writer:
            writer.release()
        cv2.destroyAllWindows()
        camera.stop() if 'camera' in dir() else None

        print("\n  Actor 정리 중...")
        for ctrl in npcs.get("ped_controllers", []):
            ctrl.stop()
        all_actors = (actor_list
                      + npcs.get("cars", [])
                      + npcs.get("peds", [])
                      + npcs.get("ped_controllers", []))
        client.apply_batch([carla.command.DestroyActor(a) for a in all_actors])
        print("  완료")


if __name__ == "__main__":
    main()
