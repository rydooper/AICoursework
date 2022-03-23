import requests
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
def outlineDetectedFace(imagePath):
    with open(imagePath, 'rb') as image_bytes:
        imageDetection = face_client.face.detect_with_stream(image=image_bytes,
                                                             return_face_id=True,
                                                             detection_model='detection_02')
        if not imageDetection:
            raise Exception('No face detected from image {}'.format(imagePath))

        # Convert width height to a point in a rectangle
        def getRectangle(faceDictionary):
            rect = faceDictionary.face_rectangle
            left = rect.left
            top = rect.top
            right = left + rect.width
            bottom = top + rect.height

            return (left, top), (right, bottom)

        def drawFaceRectangles():
            img = Image.open(imagePath)

            # draws a red box around the face
            print("See popup for results!")
            draw = ImageDraw.Draw(img)
            for eachFace in imageDetection:
                draw.rectangle(getRectangle(eachFace), outline='red')

            # Display the image in the default image browser.
            img.show()

        drawFaceRectangles()


if __name__ == '__main__':
    print("input file path: ")
    imageToGive: str = input("> ")
    # imagePath = Path(imageToGive)
    outlineDetectedFace(imageToGive)
