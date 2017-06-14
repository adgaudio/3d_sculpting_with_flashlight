"""
3d sculpting using light, webcam and a wiimote.

By: Alex Gaudio <adgaudio@gmail.com>


For really fun demonstrations with kids and curious adults.

Move my computer's mouse using the light source detected by my computer's webcam
Click the mouse using a wiimote


How to use:

    export PYTHONPATH=.:$PYTHONPATH
    sudo systemctl start bluetooth.service

    # Find the MAC address of your WiiMote and the video device you wish to use

    # push pair button on wiimote (buttons 1 and 2 together)

    python ./light_detector.py --wii_mac_addr XXX --video_device_num XXX

    An OpenCV video screen will open.  Press "w" on your keyboard to open a
    browser and start sculpting with light.
"""

import argparse as ap
import cv2
import numpy as np
from pymouse import PyMouse
from scipy.spatial.distance import euclidean
import webbrowser
import wiimote


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
    #  kernel = np.ones((10,10), np.float32) / 25
    #  img = cv2.filter2D(img,-1,kernel)
    ret, binarized = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
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
    areas = np.array([cv2.contourArea(x) for x in contours])

    centroids = [centroid(x) for x in contours]
    dist_to_last_mouse_pos = np.array([
        euclidean(x, mouse_xy_position) for x in centroids])
    ranked_contours = .2 * standard_scaler(areas) + \
        .8 * standard_scaler(1 / np.array(dist_to_last_mouse_pos))

    # hack: set hard threshold to ignore contours with medium to small area
    allowed_areas = areas > 200
    if not allowed_areas.sum() > 0:
        return
    # calculate position of the chosen contour
    i = ranked_contours[allowed_areas].argmax()

    x, y = centroids[i]
    a = areas[i]

    plot_position(img, x, y, a)
    y, x = np.array([y, x]) / img.shape
    assert x <= 1 and x >= 0
    assert y <= 1 and y >= 0

    [cv2.drawContours(img, [cv2.convexHull(x)], 0, (225, 225, 225), 2)
     for x, isok in zip(contours, allowed_areas) if isok]
    return x, y


def plot_position(img, x, y, a):
    r = min(5, int(np.sqrt(a / np.pi)))  # area of circle to its radius
    cv2.circle(img, (int(x), int(y)), 70, (255, 0, 255), thickness=10)
    cv2.circle(img, (int(x), int(y)), r, (255, 0, 255), thickness=10)


def update_mouse_position(mouse, x, y, drag, click, vertical_scroll,
                          bounding_box):
    #  w, h = mouse.screen_size()
    # operate inside a bounding box rather than whole screen
    (x1, y1), (x2, y2) = bounding_box  # top left, bottom right
    w = x2 - x1
    h = y2 - y1

    newx = int(x * w + x1)
    newy = int(y * h + y1)

    if click:
        mouse.release(*mouse.position())
        mouse.click(newx, newy)
    elif vertical_scroll:
        mouse.release(*mouse.position())
        mouse.scroll(vertical_scroll)
    elif drag:
        mouse.press(newx, newy)
    else:
        mouse.release(*mouse.position())
        mouse.move(newx, newy)


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
    w = .99
    nb = background * w + frame * (1 - w)
    return nb


def acquire_frame(cap):
    ret, frame = cap.read()
    frame = np.fliplr(frame)
    frame = frame[:, :, 0]  # grayscale on blue channel
    return frame


def main(ns):
    cap = cv2.VideoCapture(ns.video_device_num)  # creating camera object
    assert cap.isOpened(), (
        "cv2.VideoCapture(X) cannot connect with a webcam on your machine."
        " Specify a video device by number, by looking at /dev/videoX,"
        " where X is the relevant number")
    print("Hit Escape to exit")
    try:

        # initialize a mouse device
        mouse = PyMouse()
        wii = wiimote.WiiMote(ns.wii_mac_addr, 'Nintendo RVL-CNT-01')
        drag_mouse = False

        background = None
        #  last_mouse_update_time = 0
        #  last_mouse_xy = None
        #  time_delta = 0
        #  xy = None

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
            cv2.imshow('binarized', img)
            img = remove_noise_morphology(img)
            background = capture_background2(img, background)
            kernel = np.ones((25, 25))
            img = background_difference(
                img, cv2.morphologyEx(
                    background, cv2.MORPH_DILATE, kernel))
            _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
            #  img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            #  kernel = np.ones((20, 20))
            #  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            xy = select_nearest_largest_contour_centroid(
                img, mouse.position())

            #  cv2.imshow('fg', fg)
            #  cv2.imshow('binarized', binarized)

            #  l = np.zeros(img.shape + (3, ))
            #  l[:, :, 1] = img + background
            #  l[:, :, 0] = img
            #  l[:, :, 2] = background
            #  cv2.imshow('light detector', l)

            cv2.imshow('light detector', img)
            cv2.imshow('background', background)
            #  cv2.imshow('backgroundimg', img)

            if xy is not None:
                click_mouse = wii.buttons['A']
                drag_mouse = wii.buttons['B']
                vertical_scroll = \
                    (wii.buttons['Plus'] and 1) \
                    or (wii.buttons['Minus'] and -1)
                update_mouse_position(
                    mouse, *xy,
                    drag=drag_mouse, click=click_mouse,
                    vertical_scroll=vertical_scroll,
                    bounding_box=((8, 141), (1914, 990)))
            #  if xy is None:
            #  # record time
            #  time_delta = time.time() - last_mouse_update_time
            #  else:
            #  if \
            #  time_delta > 1 \
            #  and time_delta < 3 \
            #  and last_mouse_xy \
            #  and True:  #euclidean(xy, last_mouse_xy) < 200:
            #  #  and time_delta < .75 \
            #  print('click', drag_mouse)
            #  drag_mouse = True  # ~drag_mouse
            #  else:
            #  drag_mouse = False
            #  time_delta = 0
            #  last_mouse_update_time = time.time()
            #  last_mouse_xy = xy
            #  update_mouse_position(mouse, *xy, drag=drag_mouse)
            #  #  try:
            #  #  print(time_delta, euclidean(last_mouse_xy, xy))
            #  #  except:
            #  #  print('init')
            #  print('--')
    finally:
        cap.release()
        #  cv2.destroyAllWindows()


if __name__ == '__main__':

    p = ap.ArgumentParser()
    p.add_argument('--wii_mac_addr', default='EF:FF:FF:FF:7B:20')
    p.add_argument('--video_device_num', default=0, type=int,
                   help="try ls /dev/video* to get the number")

    main(p.parse_args())
