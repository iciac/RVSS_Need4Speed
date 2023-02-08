import os, glob


if __name__ == '__main__':
    trial_folder = sorted(glob.glob(os.path.join('./archive/trial*')))
    for i in range(len(trial_folder)):
        sorted(glob.glob(os.path.join(trial_folder[i], './archive/trial_*/*.jpg')))