# 🚀YOLOv5 Streamlit Deployment

---

Easy way to deploy [YOLOv5]([GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch &gt; ONNX &gt; CoreML &gt; TFLite](https://github.com/ultralytics/yolov5)) object detection model with [Streamlit](https://streamlit.io/). **feel free to edit**



Clone This repo and setup your config.

## ⚙️Config:

can be found in [app.py]([Obstacle-Detection-for-Blind-people-Deployment/app.py at main · thepbordin/Obstacle-Detection-for-Blind-people-Deployment · GitHub](https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment/blob/main/app.py))

```python
cfg_enable_url_download = True
url = "https://archive.org/download/yoloTrained/yoloTrained.pt" #Configure this if you set cfg_enable_url_download to True
cfg_model_path = "models/yourModel.pt" 
```


