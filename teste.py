import cv2
import numpy as np
import mss

with mss.mss() as sct:
    screenshot = sct.grab(sct.monitors[1])
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imshow("Screenshot", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
