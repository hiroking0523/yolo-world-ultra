import torch
import cv2
import time
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics import YOLOWorld

# ここを指定する
#############################################
# model_path = "/weight/yolo-world/yolov8x-world.pt"
model_path = "/weight/yolo-world/yolov8x-world-car-person.pt"
# model_path = "/weight/yolo-world/yolov8s-world-person.pt"
# model_path = "/weight/yolov5nu.pt"
# model_path = "/weight/yolov5su.pt"
# model_path = "/weight/yolov8n.pt"
# model_path = "/weight/yolov8s.pt"
# img_path = "/sample_data/bus.jpg"
mv_path = "/sample_data/pca_boxcam_5person.mp4"
save_output_path = "/output/sampleout.jpg"
output_video_file = "/output/yolov8x-world-person.mp4"
#############################################

# 動画ファイルを読み込む
cap = cv2.VideoCapture(mv_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps= int(cap.get(cv2.CAP_PROP_FPS))
# 保存用ファイルを準備
out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# モデルロード（cudaで）
DEVICE = torch.device('cuda')
# model = YOLOWorld(model_path).to(DEVICE)
model = YOLO(model_path).to(DEVICE)

# クラスを選ぶ
# model.set_classes(["black rabbit"])

# 推論
annotated_frames = []
total_time = 0
cnt = 0
for i in tqdm(range(1)):
    start_time = time.time()
    # results = model.predict(img_path)
    # results = model.predict(source=mv_path, show=True)
    results = model.track(source=mv_path, conf=0.3, iou=0.5, show=True, tracker="bytetrack.yaml")  # Tracking with byte tracker
    # Visualize the results on the frame
    end_time = time.time()
    total_time += end_time - start_time
    cnt += 1
    for i in range(len(results)):
        out.write(results[i].plot())

# 画像を保存
results[0].save(save_output_path)

# モデルを保存
# model.save("/weight/yolov8x-world-rabbit.pt")

# 結果
# print(f"Average inference time: {round(total_time / cnt,2)} seconds")
# print(f"FPS: {round(1 / (total_time / cnt), 2)}")

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
print('finished')