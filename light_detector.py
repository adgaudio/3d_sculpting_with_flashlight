"""
move my computer's mouse using the light source detected by my webcam

TODO: how to implement clicking?
"""
import cv2
import numpy as np
from pymouse import PyMouse
from scipy.spatial.distance import euclidean
import webbrowser

def centroid(contour):
    M = cv2.moments(contour)

    def f(n, d): return n / d if d != 0 else n
    x, y = f(M['m10'], M['m00']), f(M['m01'], M['m00'])
    return x, y


def standard_scaler(arr):
    numerator = (arr - arr.mean())
    denominator = arr.std()
    if denominator != 0:
        return numerator / denominator
    else:
        return numerator


def binarize(img):
    # #  ret, binarized = cv2.threshold(
    #     #  fg, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, binarized = cv2.threshold(img, 247, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(binarized, cv2.DIST_L2, 5)
    binarized = cv2.threshold(
        dist, 0.7 * dist.max(), 255, 0)[1].astype('uint8')
    return binarized


def remove_noise_morphology(img):
    # remove noise with an opening (erosion -> dialation)
    kernel = np.ones((5, 5))
    eroded = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    eroded = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    return eroded


def background_difference(img, background):
    img = img - (np.minimum(background, img)).astype(np.uint8)
    return img


def select_nearest_largest_contour_centroid(img, mouse_xy_position):
    # identify objects with contours (ie bounding boxes)
    i, contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return

    # pick the most relevant contour
    global areas
    areas = np.array([cv2.contourArea(x) for x in contours])

    centroids = [centroid(x) for x in contours]
    dist_to_last_mouse_pos = np.array([
        euclidean(x, mouse_xy_position) for x in centroids])
    ranked_contours = .2 * standard_scaler(areas) + \
        .8 * standard_scaler(1 / np.array(dist_to_last_mouse_pos))

    # hack: set hard threshold to ignore contours with medium to small area
    allowed_areas = areas > 1500
    if not allowed_areas.sum() > 0:
        return
    # calculate position of the chosen contour
    x, y = centroids[ranked_contours[allowed_areas].argmax()]

    plot_position(img, x, y)
    y, x = np.array([y, x]) / img.shape
    assert x <= 1 and x >= 0
    assert y <= 1 and y >= 0

    return x, y


def plot_position(img, x, y):
    cv2.circle(img, (int(x), int(y)), 70, (255, 0, 255), thickness=10)
    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), thickness=10)


def update_mouse_position(mouse, x, y, drag=False):
    w, h = mouse.screen_size()
    if drag:
        mouse.drag(int(x * w), int(y * h))
    else:
        mouse.move(int(x * w), int(y * h))


def capture_background(frames):
    print("capturing background")
    n = 20
    background = next(frames) / n
    for _ in range(n - 1):
        background += next(frames) / n
    return background


def capture_background2(frame, background):
    if background is None:
        return frame
    w = .9
    nb = background * w + frame * (1 - w)
    return nb


def acquire_frame(cap):
    ret, frame = cap.read()
    frame = np.fliplr(frame)
    frame = frame[:, :, 0]  # grayscale on blue channel
    return frame


def main():
    cap = cv2.VideoCapture(0)  # creating camera object
    assert cap.isOpened()
    print("Hit Escape to exit")

    # initialize a mouse device
    mouse = PyMouse()
    drag_mouse = False

    background = None

    try:
        background = capture_background(
            (acquire_frame(cap)) for _ in iter(int, 1))
        print('staring detector')
        while(cap.isOpened()):
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                print("ending")
                break
            elif key == ord('d'):
                drag_mouse = ~drag_mouse
            elif key == ord('w'):
                webbrowser.open('https://stephaneginier.com/sculptgl/')

            # acquire a frame from the video device
            img = acquire_frame(cap)
            cv2.imshow('frame', img)

            img = binarize(img)
            img = remove_noise_morphology(img)
            background = capture_background2(img, background)
            img = background_difference(img, background)
            xy = select_nearest_largest_contour_centroid(
                img, mouse.position())

            #  cv2.imshow('fg', fg)
            #  cv2.imshow('binarized', binarized)
            l = np.zeros(img.shape + (3, ))
            l[:, :, 1] = img + background
            l[:, :, 0] = img
            l[:, :, 2] = background
            cv2.imshow('light detector', l)
            #  cv2.imshow('light detector', img)
            #  cv2.imshow('background', background)

            if xy is None:
                continue
            update_mouse_position(mouse, *xy, drag=drag_mouse)
    finally:
        cap.release()
        #  cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
