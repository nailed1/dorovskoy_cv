import cv2

mush = cv2.imread("opencv_11_04/mushroom.jpg", cv2.IMREAD_UNCHANGED)
logo = cv2.imread("opencv_11_04/cvlogo.png", cv2.IMREAD_UNCHANGED)[:, :, :-1]

logo = cv2.resize(logo, (logo.shape[1]//2, logo.shape[0]//2))

roi = mush[:logo.shape[0], :logo.shape[1]]
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)

bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

fg = cv2.bitwise_and(logo, logo, mask=mask)

combined = cv2.add(bg, fg)
combined = cv2.addWeighted(bg, 0.7, fg, 0.3, 0)
mush[:logo.shape[0], :logo.shape[1]] = combined

print(ret)
cv2.namedWindow("Result", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("Result", mush)
cv2.waitKey(0)
cv2.destroyAllWindows()