from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
from PIL import Image
import os

project_id = '542e3f6b-c68b-4f32-986c-09371f960489'
cv_key = '27a51eaad27147bcaeb551563497d882'
cv_endpoint = 'https://ai-visionservice.cognitiveservices.azure.com/'
model_name = 'Iteration4'


def runImageClassification(folderDir):
    # Get the test images from the data/vision/test folder
    test_folder = folderDir
    test_images = os.listdir(test_folder)

    # Create an instance of the prediction service
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": cv_key})
    custom_vision_client = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)

    # Create a figure to display the results
    fig = plt.figure(figsize=(16, 8))

    # Get the images and show the predicted classes for each one
    print('Classifying images in {} ...'.format(test_folder))
    for i in range(len(test_images)):
        # Open the image and use the custom vision model to classify it
        image_contents = open(os.path.join(test_folder, test_images[i]), "rb")
        classification = custom_vision_client.classify_image(project_id, model_name, image_contents.read())
        # The results include a prediction for each tag, 0 is most probable class
        prediction = classification.predictions[0].tag_name
        # Display the image with its predicted class
        img = Image.open(os.path.join(test_folder, test_images[i]))
        threeVal: int = int(3)
        imageLen = len(test_images)
        a = fig.add_subplot(int(imageLen / threeVal), threeVal, i + 1)
        a.axis('off')
        imagePlot = plt.imshow(img)
        a.set_title(prediction)
    plt.show()


if __name__ == '__main__':
    print("Input folder of images to classify: ")
    dirFolder: str = input("> ")
    runImageClassification(dirFolder)

