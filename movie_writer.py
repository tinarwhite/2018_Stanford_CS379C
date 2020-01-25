import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

    
def visualize(state, label, stimulus):
    stimuli, temporal_points, spatial_points, clips = state
    tmax = temporal_points.shape[0]
    fig, ax = plt.subplots()
    l, = plt.plot([], [], 'k', lw=2)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    title = 'Scaled %s, Hyperparameter = %s' % (label,stimulus)
    plt.title(title,size=14)
    def update(i):
        if len(spatial_points.shape) == 1:
            x0 = spatial_points
        else:    
            x0 = spatial_points[list(stimuli).index(stimulus),i,:]
        y0 = clips[list(stimuli).index(stimulus),i,:]
        l.set_data(x0, y0)
        return l
    anim = FuncAnimation(fig, update, frames=np.arange(tmax), interval=5000/tmax)
    anim.save("visualize_%s_%s.mp4" % (label,stimulus), dpi=80, writer='ffmpeg')
    #plt.show()

def visualize2(state, label, stimulus):
	stimuli, temporal_points, spatial_points, clips = state
	tmax = temporal_points.shape[0]


	#ImageMagickFileWriter = manimation.writers['imagemagick']
	metadata = dict(title='Movie Test', artist='Matplotlib',
					comment='Movie support!')
	writer = manimation.ImageMagickWriter()

	fig = plt.figure()
	fig, ax = plt.subplots()
	title = 'Scaled %s, Hyperparameter = %s' % (label,stimuli[stimulus])
	plt.title(title,size=14)
	l, = plt.plot([], [], 'k', lw=2, animated=True)

	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	print(spatial_points.shape)
	print(clips.shape)
	print(spatial_points[stimulus,124,:])
	print(clips[stimulus,124,:])
	x0 = spatial_points[stimulus,0,:]
	#with writer.saving(fig, "visualize_%s_%s.mp4" % (label,stimulus), 200):
	for i in range(tmax):
		#x0 = spatial_points
		x0 = spatial_points[stimulus,i,:]
		y0 = clips[stimulus,i,:]
		l.set_data(x0, y0)
		#plt.plot(x0, y0, 'k', lw=2)
		writer.grab_frame()

	writer.save("visualize_%s_%s.mp4" % (label,stimulus), writer, float(tmax)/5.0, 200)