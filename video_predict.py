import os
import os.path as osp
import cv2
from PIL import Image
import streamlit as st

def runVideo(model, video):
    video_name = osp.basename(video)
    outputpath = osp.join('data/video_output', video_name)
   
    # Create A Dir to save Video Frames
    os.makedirs('data/video_frames', exist_ok=True)
    frames_dir = osp.join('data/video_frames', video_name)
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    frame_count = 0
    with st.spinner(text="Predicting..."):
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = model(frame)
            result.render()
            image = Image.fromarray(result.ims[0])
            image.save(osp.join(frames_dir, f'{frame_count}.jpg'))
        cap.release() 
        # convert frames in dir to a single video file
        os.system(f'ffmpeg -framerate 30 -i {frames_dir}/%d.jpg -c:v libx264 -pix_fmt yuv420p {outputpath}')
    # Clean up Frames Dir
    os.system(f'rm -rf {frames_dir}')
    

    # Display Video
    output_video = open(outputpath, 'rb')
    output_video_bytes = output_video.read()
    st.video(output_video_bytes)
    st.write("Model Prediction")
    