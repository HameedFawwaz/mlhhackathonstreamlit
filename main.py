import streamlit as st
from pathlib import Path as path
from moviepy.editor import VideoFileClip
from processes.process import process

st.title("Deep Fake Generator")
st.header("Create deep faked videos from images using templates!")

st.write("Welcome to the deep fake generator! Here are the instructions to use the deep fake generator!\n 1. Upload the image you want to deep fake! (accepted formats: PNG, JPG)\n 2. Upload the template video you would like to use, it must be less than 30 seconds in length (accepted format: MP4)\n 3. Press submit and please wait up to 20 minutes for your request to process and do not leave the application!")

with st.form(key="Form :", clear_on_submit= True):

    img = st.file_uploader(label="upload the image you want to deepfake")
    temp = st.file_uploader(label="upload the template you would like to use")
    submit = st.form_submit_button(label="Submit")

if submit:

    st.write("Processing... Please allow up to 15 minutes for your request to process.")
    img_path = path("./saved", img.name)
    temp_path = path("./saved", temp.name)
    with open(img_path, mode="wb") as w:
        w.write(img.getvalue())
    with open(temp_path, mode="wb") as w:
        w.write(temp.getvalue())
        

    if img_path.exists() and temp_path.exists():  

        len_vid = VideoFileClip(f"./saved/{temp.name}").duration
        st.write("video length: " + str(len_vid))

        if len_vid <= 30:
            process.generate(img.name, temp.name)

            img_name = img.name.partition(".")
            video_file = open(f'./generated/{img_name}.mp4', "rb")
            video_bytes = video_file.read()

            st.video(video_bytes)
        else:
            st.write("Error! Make sure that your template video is less than 30 seconds in length!")

