import requests
import subprocess
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices import vision
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# authentication
KEY = "bb56758e1ec6467c8983192b3d667853"
ENDPOINT = "https://ai-face-recognition.cognitiveservices.azure.com/"
face_client = vision.face.FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))


# Detect a face in an image that contains a single face
def outlineDetectedFace(imageGiven):
    imageDetection = face_client.face.detect_with_stream(image=imageGiven,
                                                         return_face_id=True,
                                                         detection_model='detection_03')
    '''single_face_image_url = 'https://raw.githubusercontent.com/rydooper/supernatural-images-dataset/main' \
                            '/Supernatural_Train_Dataset/Dean/Dean0112.jpg'
    single_image_name = os.path.basename(single_face_image_url)
    # We use detection model 3 to get better performance.
    detected_faces = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_03')'''
    if not imageDetection:
        raise Exception('No face detected from image {}'.format(imageGiven))

    # Convert width height to a point in a rectangle
    def getRectangle(faceDictionary):
        rect = faceDictionary.face_rectangle
        left = rect.left
        top = rect.top
        right = left + rect.width
        bottom = top + rect.height

        return (left, top), (right, bottom)

    def drawFaceRectangles():
        # Download the image from the url
        response = requests.get(imageGiven)
        img = Image.open(BytesIO(response.content))

        # For each face returned use the face rectangle and draw a red box.
        print('Drawing rectangle around face... see popup for results.')
        draw = ImageDraw.Draw(img)
        for eachFace in imageDetection:
            draw.rectangle(getRectangle(eachFace), outline='red')

        # Display the image in the default image browser.
        img.show()

    drawFaceRectangles()


if __name__ == '__main__':
    print("input file path: ")
    imageToGive = input("> ")
    outlineDetectedFace(imageToGive)
