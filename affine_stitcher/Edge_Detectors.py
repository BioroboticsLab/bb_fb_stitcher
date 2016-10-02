import cv2

def dog(img, sigma1, sigma2):
    g1 = cv2.GaussianBlur(img, (0,0), sigma1)
    g2 = cv2.GaussianBlur(img, (0,0), sigma2)
    return g1 - g2

