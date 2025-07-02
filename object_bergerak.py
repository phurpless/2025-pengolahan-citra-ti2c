import cv2
import imutils
import numpy as np
from imutils.perspective import order_points

KNOWN_WIDTH_CM = 2.0
pixels_per_cm = None

def measure_object(frame, cnt, pixels_per_cm):
    if cv2.contourArea(cnt) < 100:
        return frame

    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = order_points(box)

    (tl, tr, br, bl) = box
    width = np.linalg.norm(tr - tl)
    height = np.linalg.norm(br - tr)

    dimA = height / pixels_per_cm
    dimB = width / pixels_per_cm

    cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.putText(frame, "{:.2f} cm".format(dimA),
                (int(tl[0]), int(tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)
    cv2.putText(frame, "{:.2f} cm".format(dimB),
                (int(tr[0] + 10), int(tr[1])), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

    return frame

# Buka video
cap = cv2.VideoCapture("D:/Semester 4/Pak Prayit Pengolahan Citra/Pengukuran Objek Bergerak/object_bergerak.mp4")
if not cap.isOpened():
    print("[ERROR] Video tidak ditemukan atau gagal dibuka.")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video selesai atau tidak bisa dibaca.")
        break

    frame = imutils.resize(frame, width=800)
    fgmask = fgbg.apply(frame)

    # Debug: tampilkan mask deteksi gerakan
    cv2.imshow("Mask Gerakan", fgmask)

    # Proses mask untuk hilangkan noise
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Ambil kontur gerakan
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:  # Supaya tidak deteksi noise
            if pixels_per_cm is None:
                box = cv2.minAreaRect(largest)
                width_in_pixels = box[1][0]
                if width_in_pixels > 0:
                    pixels_per_cm = width_in_pixels / KNOWN_WIDTH_CM
                    print(f"[INFO] Kalibrasi: {pixels_per_cm:.2f} pixels/cm")

            if pixels_per_cm:
                frame = measure_object(frame, largest, pixels_per_cm)

    cv2.imshow("Hasil Pengukuran", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()