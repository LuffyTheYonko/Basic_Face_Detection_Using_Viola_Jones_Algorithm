import cv2
import streamlit as st

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')


def detect_faces(scalefactor, neighbors, color):
    # initialize the webcam

    cap = cv2.VideoCapture(0)

    while True:
        # read the frames from the webcam

        ret, frame = cap.read()

        # saving captured image
        filename = "SavedImage.jpg"

        cv2.imwrite(filename, frame)

        # convert the frames to greyscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect the faces using the face cascade classifier

        faces = face_cascade.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=neighbors)

        # convert color to BGR
        bgr_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

            # Display the frames

            cv2.imshow("Face Detection using Viola-Jones Algorithm", frame)

            # Exit the loop when "q" is pressed

        if cv2.waitKey(0) & 0xFF == ord("q"):

            break



    # Release the webcam and close all windows

    cap.release()
    cv2.destroyAllWindows()

    print("Done here")


def app():
    st.title("Face detection using Viola-Jones Algorithm")
    st.write("How to use this app:")
    st.write("Use the color picker tool to choose the border color around your face.")
    st.write("Use the scale slider to to select how thick you want your facial borders to be.")
    st.write("Use the neighbors slider to select how close you want your facial borders to be.")
    st.write("Click the detect button and watch the magic happen.")

    st.write("Press the button below to start detecting faces from your webcam")

    color = st.color_picker("Detection Border Color", '#00f900', key="color")

    scalefactor = st.slider("Select ScaleFactor", 0.0, 2.0, 1.3)

    neighbors = st.slider("Select minNeighbors", 0, 6, 5)
    # add a button to start detecting faces

    if st.button("Detect Faces"):
        detect_faces(scalefactor, neighbors, color)


if __name__ == "__main__":
    app()
