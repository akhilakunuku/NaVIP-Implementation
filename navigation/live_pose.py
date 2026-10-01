import cv2
import torch
from PIL import Image

from pose_inference import transform, model, DEVICE


print("Starting webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam opened successfully.")
print("Press Q to stop.")


frame_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    frame_count += 1

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL image
    image = Image.fromarray(
        rgb_frame
    )

    # Preprocess
    input_tensor = transform(
        image
    )

    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # Prediction
    with torch.no_grad():

        predicted_position, predicted_orientation = model(
            input_tensor
        )

    # Normalize quaternion
    predicted_orientation = (
        predicted_orientation /
        (
            torch.norm(
                predicted_orientation,
                dim=1,
                keepdim=True
            ) + 1e-8
        )
    )

    position = (
        predicted_position[0]
        .cpu()
        .numpy()
    )

    orientation = (
        predicted_orientation[0]
        .cpu()
        .numpy()
    )

    # Display
    text1 = (
        f"Position: "
        f"{position[0]:.2f}, "
        f"{position[1]:.2f}, "
        f"{position[2]:.2f}"
    )

    text2 = (
        f"Quaternion: "
        f"{orientation[0]:.2f}, "
        f"{orientation[1]:.2f}, "
        f"{orientation[2]:.2f}, "
        f"{orientation[3]:.2f}"
    )

    text3 = f"Frames: {frame_count}"

    cv2.putText(
        frame,
        text1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        text2,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        text3,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "NaVIP Live Pose",
        frame
    )

    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(
            f"Processed {frame_count} camera frames"
        )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()

print()
print("======================================")
print("Live Pose Test Completed")
print("======================================")
print(
    "Total frames processed:",
    frame_count
)
print("======================================")