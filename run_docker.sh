# Xサーバーへのアクセスを現在のユーザーに限定して許可する
xhost +

# Dockerコンテナを起動し、現在のユーザーのDISPLAY環境変数とX11のソケットを渡す
sudo docker run --gpus all -it --rm \
  --runtime nvidia \
  --shm-size=1g \
  -v ~/workspase/yolo-world-ultra/test.py:/usr/src/ultralytics/test.py \
  -v ~/workspase/yolo-world-ultra/test_nishi.py:/usr/src/ultralytics/test_nishi.py \
  -v ~/workspase/yolo-world-ultra/weight:/weight \
  -v ~/workspase/yolo-world-ultra/sample_data:/sample_data \
  -v ~/workspase/yolo-world-ultra/output:/output \
  -e DISPLAY=$DISPLAY \
  --network host \
   yolo-world-ultra:latest