import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
from processes.demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
from moviepy.editor import *
from moviepy.editor import VideoFileClip

class process():

    def generate(img, temp):
        source_image = imageio.imread(f"./saved/{img}")

        #unedited_vidclip = VideoFileClip(f"./template/{temp}")
        

        reader = imageio.get_reader(f"./saved/{temp}")
        
        
        source_image = resize(source_image, (256, 256))[..., :3]

        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        generator, kp_detector = load_checkpoints(config_path='./processes/config/vox-256.yaml', 
                                checkpoint_path='./processes/config/vox-cpk.pth.tar')

        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

        img = img.partition(".")

        imageio.mimsave(f'./generated/{img}.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)

        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=True)

        #vidclip = VideoFileClip(f"./template/")
        #audclip = ffmpeg.input(f"./processes/template/{temp}.mp3")

       # out = ffmpeg.output(vidclip, audclip, f"./generated/{img}2.mp4")
        #out.run()

    
