# AICoursework
This is the coursework for the Artificial Intelligence module.
There are two chatbots included, in their separate folders.

Chatbot 1 satisifies part a and b, while Chatbot 2 satisfies part c and d, while inheriting the features from part a and b.

A) Rule-based and similarity-based conversation features. The chatbot can respond to the user, based on an AIML file and utilising pre-defined question and answer pairs. The similarity between a user input and the response required is calculated using a bag-of-words, tf/idf and cosine similarity. BONUS FEATURE: speech-based input from the user is provided for. 

B) Logical reasoning features. A simple first order logic knowledgebase and inference engine using the Natural Language Toolkit library. The user may check that a fact is true, update the knowledge base with "I know that" phrases or even evaluate the knowledge base to ensure that no information is contradictory. BONUS FEATURE: fuzzy logic implementation that allows for episodes of the topic (Supernatural) to be rated and an average calculated.

C) Local image classification model. A user can supply the filepath to any locally downloaded image and have it classified as either: Sam Winchester, Cas or Dean Winchester. The chatbot is trained on images of these three characters from the Supernatural show with a h5 file, containing a Convolutional Neural Network (CNN) model. There is no bonus feature for this part.

D) Cloud-based image classification service. A user can choose to use the Azure-based classification AI model to identify an image (or set of images) as either: Sam Winchester, Dean Winchester, Castiel or "Negative" (none of them). Sadly, my subscription to Azure has since ended so this no longer works. BONUS FEATURE: Azures' face recognition features implemented.
