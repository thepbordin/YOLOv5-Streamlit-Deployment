# 🚀YOLOv5 Streamlit Deployment

Easy way to deploy [YOLOv5](https://github.com/ultralytics/yolov5) object detection model with [Streamlit](https://streamlit.io/). 



**Please feel free to use/edit.** 

codes modified by GitHub/thepbordin from GitHub/zhoroh

Clone This repo and read install instruction from [Installation Guide](https://github.com/thepbordin/YOLOv5-Streamlit-Deployment#%EF%B8%8Finstallation-guide).



## ✨Features

- Support large model file. (Don't require GitLFS; Download model from URL)

- Support small model file. (<100MB; uploaded to github repository)

- Select images randomly by slider from test set

- Upload image/video

- Select computing device (cuda/cpu)



## ⚙️Installation Guide

configs can be found in [app.py](https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment/blob/main/app.py)

### Download model from URL:

1. Upload model to [Internet Archive](https://archive.org/)

2. Go to your uploaded file page.

3. From "DOWNLOAD OPTIONS" select "SHOW ALL".

4. Right click at <yourmodelname>.pt and Copy link address.

5. Edit config in [app.py](https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment/blob/main/app.py)
   
   - cfg_enable_url_download = True
   
   - url = "<copied link address>"

```python
cfg_enable_url_download = True
url = "your_model_url"
```

### Use local .pt file:

Edit config in [app.py](https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment/blob/main/app.py)

```python
## CFG
cfg_model_path = "models/your_model_name.pt" 
```


