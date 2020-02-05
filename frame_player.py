from matplotlib import animation, rc
from JSAnimation.IPython_display import display_animation
from IPython.display import display
from IPython.display import HTML
import os
import glob
import io
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