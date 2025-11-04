import cv2
import time
import os

# Zapytaj użytkownika, czy chce podać adres strumienia ręcznie
# INPUT_URL = input("Do you want to input the stream URL manually? (y/n): ").lower() == "y"

INPUT_URL = False

# Adres strumienia z telefonu (np. IP Webcam)
if INPUT_URL:
    stream_url = input("Stream URL: ")
    stream_port = input("Stream Port: ")
else:
    stream_url = "10.152.79.206"
    stream_port = "4747"
stream_string = f"http://{stream_url}:{stream_port}/video"
cap = cv2.VideoCapture(stream_string)


if not cap.isOpened():
    print("Can't open stream. Check the URL and/or Wi-Fi connection.")
    exit()

frame_id = 0
os.makedirs("data", exist_ok=True)
save_path = f"{os.getcwd()}/data/"
interval = 1  # co ile sekund zapisywać klatkę
crop_top = 20  # ile pikseli przyciąć od góry

last_save = time.time()

print("Opening stream. Press 'q' to quit, 's' to save frame.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read or error occurred. Exiting...")
        break

    # Przycinanie górnej części klatki
    frame = frame[crop_top:, :]

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # naciśnięcie 's' zapisuje pojedynczą klatkę
    if key == ord("s"):
        filename = f"{save_path}frame_{frame_id:05d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        frame_id += 1

    # automatyczny zapis co określony czas (opcjonalnie)
    if False:
        # if time.time() - last_save > interval:
        filename = f"{save_path}auto_{frame_id:05d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Auto save: {filename}")
        frame_id += 1
        last_save = time.time()

    if key == ord("q"):
        print("Exit requested. Stopping...")
        break

cap.release()
cv2.destroyAllWindows()
