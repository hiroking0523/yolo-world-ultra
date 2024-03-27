# test nishi
import cv2
import time
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics import YOLOWorld


def save_custom_model(model_path: str, save_path: str, class_names: list):
    # load model
    model = YOLOWorld(model_path)
    # Define custom class names
    model.set_classes(class_names)
    # Save model
    model.save(save_path)


def inference_benchmark(model_path: str, img_path: str, data_type: str, save_output_path: str = None, device: str = "cpu"):
    """ Benchmark inference time """
    import torch
    DEVICE = torch.device('cuda')
    # DEVICE = torch.device('cpu')
    
    # Load model
    model = YOLOWorld(model_path).to(DEVICE)
    # Define custom classes
    # model.set_classes(["wombat"])

    if data_type == "image":
        total_time = 0
        cnt = 0
        for i in tqdm(range(30)):
            # Load image
            img = cv2.imread(img_path)
            # Inference
            start_time = time.time()
            # data to device
            with torch.no_grad():
                # results = model.predict(img, 
                #                         device=DEVICE,
                #                         save=False,
                #                         verbose=False)
                results = model(img, 
                                device=DEVICE,
                                save=False,
                                verbose=False)
            end_time = time.time()
            
            total_time += end_time - start_time
            cnt += 1
        
        print(f"Average inference time: {round(total_time / cnt,2)} seconds")
        print(f"FPS: {round(1 / (total_time / cnt), 2)}")
        results[0].show()
        # Save output
        if save_output_path:
            results[0].save(save_output_path)
    
    elif data_type == "video":
        # Load video
        vid = cv2.VideoCapture(img_path)
        
        # Inference
        cnt = 0
        total_time = 0

        if save_output_path:
            out = cv2.VideoWriter(save_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            start_time = time.time()
            results = model(frame)
            end_time = time.time()
            # print(f"Inference time: {end_time - start_time} seconds")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Save output
            if save_output_path:
                out.write(results[0].plot())
            cnt += 1
            total_time += end_time - start_time
        
        print(f"Average inference time: {total_time / cnt} seconds")
        print(f"FPS: {round(1 / (total_time / cnt), 2)}")
        vid.release()
        # if save_output_path:
        #     for i in range(len(results)):
        #         out.write(results[i].plot())
        #     out.release()


if __name__ == "__main__":
    model_path = "/weight/yolov8s-world-person.pt"
    img_path = "/sample_data/bat.png"
    mv_path = "/sample_data/person_fp30.mp4"
    save_output_path = "/output/sampleout_bat.png"
    output_video_file = "/output/test_tanoue.mp4"
    # img_path = "/sample_data/person_fp30.mp4"
    # save_output_path = "/output/sampleout_video.mp4"
    inference_benchmark(model_path, mv_path, "video", output_video_file, "cuda")
