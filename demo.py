import cv2
import numpy as np
from inference import SegApp

INTERESTED_CLASS = 4


def main():
    app = SegApp()
    app.spin()
    img = cv2.imread('demo.jpg')
    _ = app.process(img)

    mask = app.get_result()[0]
    mask[mask == [INTERESTED_CLASS]] = 255

    mask = np.array(mask, dtype=np.uint8)
    cv2.imshow('res', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
