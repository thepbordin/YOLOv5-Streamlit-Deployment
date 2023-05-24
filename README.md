
# 🚀 YOLOv5 Streamlit Deployment
[![HitCount](https://hits.dwyl.com/thepbordin/YOLOv5-Streamlit-Deployment.svg?style=flat&show=unique)](http://hits.dwyl.com/thepbordin/YOLOv5-Streamlit-Deployment)


A Easy way to deploy [YOLOv5](https://github.com/ultralytics/yolov5) object detection model with [Streamlit](https://streamlit.io/). 

**Please feel free to use/edit.** 


code modified by GitHub/thepbordin from GitHub/zhoroh

## ✨ Features

- YOLO Weights Source
	-  Load from Local
	- Download Weights from URL
- Example Dataset 
  - Videos
  -	 Images
 - Upload Data 
   - Video
   - Image
- Select computing device (cuda/cpu)



## ⚙️ Installation



### Local Use
1. Install Requirements 
	`pip install -r requirements.txt`
2. Install ffmpeg (for video inferencing)
	- For Windows [read here](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
	- For Mac (brew)
		`brew install ffmpeg`
3. Strart Stremlit
	```
	cd YOLOv5-Streamlit-Deployment
    streamlit run app.py
    ```
### Streamlit Cloud
1. Edit a configuration in app.py (read ⚙️ Config Instruction)
2. (Optional) Upload example datas in
	- `example_images`
	- `example_videos`
4. Deploy on [Streamlit](https://share.streamlit.io/deploy)


## ⚙️ Config Instruction
### Download model from URL

1. Upload model to [Internet Archive](https://archive.org/)
2. Go to your uploaded file page.
3. From `DOWNLOAD OPTIONS` select `SHOW ALL`
4. Right click at <yourmodelname>.pt and Copy link address.
5. Edit config in [app.py](https://github.com/thepbordin/YOLOv5-Streamlit-Deployment/blob/main/app.py)

	```python
	cfg_enable_url_download = True
	url = "your_model_url"
	```

### Use local .pt file:
Edit config in [app.py](https://github.com/thepbordin/YOLOv5-Streamlit-Deployment/blob/main/app.py)
```python
## CFG
cfg_model_path = "models/your_model_name.pt" 
```

## Reference
[Yolov5 Real-time Inference using Streamlit](https://github.com/moaaztaha/Yolo-Interface-using-Streamlit)
