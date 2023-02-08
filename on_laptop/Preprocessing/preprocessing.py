import cv2
import imageio.v2 as imageio
import os

from tqdm import tqdm

stem = '/home/pi/RVSS_Need4Speed/on_robot/collect_data/archive/'


def make_gif(folder_path, gif_name):
    images = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image = imageio.imread(os.path.join(folder_path, filename))
            images.append(image)
    imageio.mimsave(gif_name, images, fps=10)


def threshold_image(image_path, threshold, max_val):
    image = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(image, threshold, max_val)
    return thresh




if __name__ == "__main__":
    make_gif(stem+'trial_01', 'trial_01.gif')
