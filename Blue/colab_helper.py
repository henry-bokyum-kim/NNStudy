
# coding: utf-8

# <a href="https://colab.research.google.com/github/henry-bokyum-kim/NNStudy/blob/master/GymDisplay.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Turn on virtual display

get_ipython().system('pip install gym pyvirtualdisplay > /dev/null 2>&1')
get_ipython().system('apt-get update ')
get_ipython().system('apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1')
get_ipython().system('apt-get install x11-utils')

from pyvirtualdisplay import Display
from IPython import display as ipythondisplay

v_display = Display(visible=0, size=(1400,900),)
v_display.start()


# # Show frame using HTML player

get_ipython().system('pip install JSAnimation')
from matplotlib import animation
from JSAnimation.IPython_display import display_animation
from IPython.display import display
from IPython.display import HTML
import matplotlib.pyplot as plt

# Imports specifically so we can render outputs in Colab.
def display_frames_as_gif(frame, intv=30):
    """Displays a list of frames as a gif, with controls."""
    fig = plt.figure()
    patch = plt.imshow(frame[0].astype(int))
    def animate(i):
        patch.set_data(frame[i].astype(int))
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frame), interval=intv, blit=False
    )
    #display(display_animation(anim, default_mode='loop'))
    # Set up formatting for the movie files
    display(HTML(data=anim.to_html5_video()))
    #FFwriter = animation.FFMpegWriter()
    #anim.save('basic_animation.mp4', writer = FFwriter)
    #show_video()
# display 

display_frames_as_gif(frame)


# # Show frame using gym.Monitor

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        for mp4 in mp4list:
            video = io.open(mp4, 'r+b').read()
            print(mp4)
            encoded = base64.b64encode(video)
            ipythondisplay.display(HTML(data='''<video alt="test" autoplay loop controls style="height: 300px;">
                                      <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                                      </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


# # Box2d Install

get_ipython().system('apt-get install swig')
get_ipython().system('pip3 install box2d box2d-kengz')


# # GDrive Save

from google.colab import drive 
drive.mount('/content/gdrive/')

import pickle
with open('/content/gdrive/My Drive/noise','rb') as f:
    noise = pickle.load(f)

