import cv2
from inference import SegApp

INTERESTED_CLASS = 4


def main():
    app = SegApp(model_name='PSPNet101')
    app.spin()
    img = cv2.imread('demo.jpg')
    res = app.process(img)

    # Uncomment below four lines of code to visualize the class of interests, aka, 4
    # import numpy as np
    # mask = app.get_result()[0]
    # mask[mask == [INTERESTED_CLASS]] = 255
    # res = np.array(mask, dtype=np.uint8)

    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
