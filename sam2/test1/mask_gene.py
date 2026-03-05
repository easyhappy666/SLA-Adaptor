import numpy as np
import cv2
from scipy.ndimage import label
import random

def msp_line(W=256, H=256):
    M = np.zeros((H, W), dtype=np.uint8)
    c = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    angle = np.random.uniform(0, 180)
    theta = np.deg2rad(angle)
    l = np.random.uniform(60, 200)
    s = np.random.randint(20, 40)
    x = np.linspace(-l/2, l/2, s)
    y = np.zeros_like(x)
    if np.random.rand() < 0.5:
        eps = np.random.normal(0, 14, size=s).astype(np.float32).reshape(-1, 1)
        y += cv2.GaussianBlur(eps, (1, 5), 2).flatten()
    min_thickness, max_thickness = 1, 7
    thickness_curve = np.linspace(min_thickness, max_thickness, s // 2).tolist() + \
                      np.linspace(max_thickness, min_thickness, s - s // 2).tolist()
    thickness_curve = np.array(thickness_curve) + np.random.uniform(-0.5, 0.5, size=s)
    thickness_curve = np.clip(thickness_curve, min_thickness, max_thickness)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
    for i in range(s - 1):
        pt1 = (R @ np.array([x[i], y[i]]) + c).astype(int)
        pt2 = (R @ np.array([x[i+1], y[i+1]]) + c).astype(int)
        cv2.line(M, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, int(thickness_curve[i]), cv2.LINE_AA)
    _, M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)
    return M

def msp_dot(W=256, H=256):
    M = np.zeros((H, W), dtype=np.uint8)
    c = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    r = np.random.uniform(5, 35)
    s = np.random.randint(12, 30)
    theta = np.sort(np.random.uniform(0, 2*np.pi, s))
    alpha, beta = np.random.uniform(0.6, 1.4), np.random.uniform(0.05, 0.35)
    r_i = np.random.uniform(-beta*r, beta*r, s) if np.random.rand() >= 0.66 else np.random.normal(0, beta*r, s)
    x = c[0] + (r + r_i) * np.cos(theta) * alpha
    y = c[1] + (r + r_i) * np.sin(theta)
    contour = np.stack((x, y), axis=1).astype(np.int32)
    cv2.fillPoly(M, [contour], 255)
    if np.random.rand() < 0.5:
        k = np.random.choice([3, 5, 7])
        M = cv2.GaussianBlur(M, (k, k), 0)
    _, M = cv2.threshold(M, 127, 255, cv2.THRESH_BINARY)
    return M

def msp_freeform(W=256, H=256):
    M = np.zeros((H, W), dtype=np.uint8)
    n_step = np.random.randint(300, 18001)
    sigma = np.random.uniform(2, 12)
    x, y = np.random.randint(0, W), np.random.randint(0, H)
    for _ in range(n_step):
        M[y, x] = 1
        x = np.clip(x + np.random.choice([-1, 0, 1]), 0, W - 1)
        y = np.clip(y + np.random.choice([-1, 0, 1]), 0, H - 1)
    ksize = int(2 * sigma + 1) | 1
    M = cv2.GaussianBlur(M.astype(np.float32), (ksize, ksize), sigmaX=sigma)
    _, M = cv2.threshold(M, 0.5, 1, cv2.THRESH_BINARY)
    labeled, num = label(M, structure=np.ones((3, 3)))
    if num > 1:
        M = (labeled == (1 + np.argmax([(labeled == i).sum() for i in range(1, num + 1)]))).astype(np.uint8)
    return M.astype(np.uint8) * 255


import numpy as np
import random
import cv2


def generate_meta_mask(W=256, H=256, m_max=2, shape_types=None):


    fn_map = {
        "line": msp_line,
        "dot": msp_dot,
        "freeform": msp_freeform
    }


    if shape_types is None or len(shape_types) == 0:

        available_fns = [msp_line, msp_dot, msp_freeform]
    else:
        available_fns = []
        for name in shape_types:
            if name in fn_map:
                available_fns.append(fn_map[name])

        if not available_fns:
            available_fns = [msp_line, msp_dot, msp_freeform]

    m = np.random.randint(1, m_max + 1)
    mask_final = np.zeros((H, W), dtype=np.uint8)

    for _ in range(m):

        fn = random.choice(available_fns)

        temp_mask = fn(W=256, H=256)
        temp_mask = cv2.resize(temp_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        mask_final = cv2.bitwise_or(mask_final, temp_mask)

    return mask_final