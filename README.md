# ReelTime_FaceRecognition
This project provide an usefull case of the <b>LBPH</b> (Local Binary Patterns Histograms) face recognition method implemented in *OpenCV*.
The model try to predict in reel time the gender (male or female) of a person from its facial attribute.

Requirements:
- OpenCV (version 4.2 used)
- QT4 (not 5 or higher)
- A webcam 

The model was trained using 47 images (19 women and 28 men) and the testing to evaluate our model was done in a set of 30 images (11 wowen and 19 men).
The model is stored in __data/lbph_model.yml__ file.

Here we use our model in a real time process. For that we combine it with our Camera and as soon as a capture is made, we call the __detectAndCut__ function on our image. This function uses the OpenCV functions (__face_cascade__ and __eyes_cascade__ with the __XML__ classifiers that go well) to detect a face and measure the position of its eyes. Once the coordinates are obtained, we call the Python script __*crop_face.py*__ (see appendix 2) which aligns, scales and trims the image on the face to be studied. It is indeed important to obtain an image of the same size as those in our database.
Then simply use our model to predict the outcome on the image obtained and display the result in our window.
The interface is done with *QT4*. 

You can check the result on the video provided. 




