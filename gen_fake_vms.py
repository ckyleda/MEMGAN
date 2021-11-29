import numpy as np
from random import randint


# Randomly select tl, br corners
def gen_images(batch_size):
    batch_labels = []
    for x in range(batch_size):
        image = np.zeros((10, 10), dtype='float32')
        tl1 = randint(0, 9)
        tl2 = randint(0, 9)
        x_dist = randint(tl1+1, 10)
        y_dist = randint(tl2+1, 10)

        step1_intensity = randint(1, 255)
        image[tl1:x_dist, tl2:y_dist] = step1_intensity

        # Do another step
        if randint(0, 1) == 1:
            s2_pt1 = randint(tl1, x_dist)
            s2_pt2 = randint(tl2, y_dist)

            s2_x_dist = randint(s2_pt1, x_dist)
            s2_y_dist = randint(s2_pt2, y_dist)

            image[s2_pt1:s2_x_dist, s2_pt2:s2_y_dist] = randint(step1_intensity, 255)


        image = image / 255.0


        batch_labels.append(image.flatten())
    return np.array(batch_labels)

