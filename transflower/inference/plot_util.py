import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation

def save_poses(out_dir, poses, prefix=""):
#     date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
    out_filename = out_dir + prefix  + ".npz"
    os.makedirs(out_dir, exist_ok=True)
    #print("save ", len(poses)," poses to file ",out_filename)
    with open(out_filename, "wb") as out_file:
        np.save(out_file, poses)
        
def is_iteratable(data):
    try:
        if hasattr(data, "__getitem__"):
            some_object_iterator = iter(data)
            return True
    except TypeError as te:
        pass
    return False

def print_hierarchy(data, level=0):
    tab_string = "\t"*level
    try:
        for key in data:
            print(tab_string,data[key])
            if is_iteratable(data[key]):
                print_hierarchy(data[key],level+1)
    except:
        pass
    return




def draw_pose(pose, center=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = np.array(pose).T
    print(p.shape)
    ax.scatter(p[0,:], p[2,:], p[1,:])
    if center:
        ax.set_xlim3d([-100.0, 100.0])

        ax.set_ylim3d([-100.0, 100.0])

        ax.set_zlim3d([-100.0, 100.0])
    plt.show()
    return


def draw_poses(poses, center=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in poses:
        p = np.array(p).T
        ax.scatter(p[0,:], p[2,:], p[1,:])
    if center:
        ax.set_xlim3d([-100.0, 100.0])

        ax.set_ylim3d([-100.0, 100.0])

        ax.set_zlim3d([-100.0, 100.0])
    plt.show()
    return

def draw_pose_sequences(sequences, step_size, center=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for seq in sequences:
        x,y,z = [], [], []
        for i in range(0, len(seq), step_size):
            pose = np.array(seq[i]).T
            x += pose[0].tolist()
            y += pose[2].tolist()
            z += pose[1].tolist()
        ax.scatter(x,y,z)
    if center:
        ax.set_xlim3d([-100.0, 100.0])

        ax.set_ylim3d([-100.0, 100.0])

        ax.set_zlim3d([-100.0, 100.0])
    plt.show()
    return

def update_pose_plot(num, ax, poses, center):
    p = poses[num]
    ax.clear()
    ax.scatter(p[0,:], p[2:], p[1,:])
    
    if center:
        # Setting the axes properties
        ax.set_xlim3d([-100.0, 100.0])

        ax.set_ylim3d([-100.0, 100.0])

        ax.set_zlim3d([-100.0, 100.0])

    ax.text(100,-100,0, "frame"+str(num), size=10, zorder=1)
    return

def draw_animation(poses, use_jupyter=False, center=True, fps=60):
    """ https://stackoverflow.com/questions/38118598/3d-animation-using-matplotlib """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

    N = len(poses)
    anim = animation.FuncAnimation(fig, update_pose_plot, N, fargs=(ax, poses, center), interval=1000/fps, blit=False)

    if not use_jupyter:
        plt.show()
    else:
        from IPython.display import HTML
        return HTML(anim.to_html5_video())
        

def extract_pose(data, n_points=None, o=3):
    points = []
    if n_points is None:
        n_points = len(data)
    while o < n_points:
        points.append(data[o:o+3])
        o += 3
    return points


def update_samples_plot(num, ax, samples, center):
    ax.clear()
    for poses in samples:
        p = poses[num]
        ax.scatter(p[0,:], p[2:], p[1,:])
    
    if center:
        # Setting the axes properties
        ax.set_xlim3d([-100.0, 100.0])

        ax.set_ylim3d([-100.0, 100.0])

        ax.set_zlim3d([-100.0, 100.0])

    ax.text(100,-100,0, "frame"+str(num), size=10, zorder=1)
    return

def draw_samples(samples, use_jupyter=False, center=True):
    """ https://stackoverflow.com/questions/38118598/3d-animation-using-matplotlib """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

    N = len(samples[0])
    anim = animation.FuncAnimation(fig, update_samples_plot, N, fargs=(ax, samples, center), interval=1000/N, blit=False)

    if not use_jupyter:
        plt.show()
    else:
        from IPython.display import HTML
        HTML(anim.to_html5_video())

