
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <ctime>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
/** Function Headers */
void detectAndCut(cv::Mat frame);

/** Global variables */
std::string face_cascade_name = "data/haarcascade_frontalface_alt.xml";
std::string eyes_cascade_name = "data/haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;
std::string window_name = "Capture - Face detection";
cv::RNG rng(12345);
std::string machaine;
bool ok = false;

int main(int argc, char *argv[])
{
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    cv::FileStorage model_file("data/trainningData.yml", cv::FileStorage::READ);
    std::cout<<"test"<<std::endl;
    std::cout<<"model created"<<std::endl;
    model->read("data/lbph_model.yml");
    //model->load<cv::face::FaceRecognizer>("data/trainningData.yml");
    std::cout<<"model charged"<<std::endl;

    cv::VideoCapture cap(0); // open the default camera

    if(!cap.isOpened())      // check if we succeeded
    {
        std::cerr<<"Failed to open Camera"<<std::endl;
        return 1;
    }

    int cpt = 0;
    int predictedLabel = -1;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    while(true)
    {
        cpt++;
        cv::Mat image;
        cap >> image;        // read an image from the webcam
        cv::imwrite("data/image.jpg",image);

        if(! image.data )    // Check for invalid input
        {
            std::cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        if( !image.empty() && cpt%20 == 0)
        {
            detectAndCut(image);
            for(int i = 0 ; i < 1000 ; i++);
            cv::Mat image_cropped;
            image_cropped = cv::imread("image_cropped.jpg",0);
            if(ok == true)
            {
                predictedLabel = model->predict(image_cropped);
                //std::cout<< "Predicted label = " << predictedLabel << std::endl;
                ok = false;
            }
            else
            {
                predictedLabel = -1;
            }

        }
        else if (image.empty())
        {
            printf(" --(!) No captured frame -- Break!");
        }

        namedWindow(window_name, cv::WINDOW_AUTOSIZE);// Create a window for display.

        if(predictedLabel == -1) machaine = "Pas de visage";
        if(predictedLabel == 0) machaine = "Femme";
        if(predictedLabel == 1) machaine = "Homme";

        cv::putText(image,machaine,cv::Point(50,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(47,79,79),4);
        cv::imshow(window_name, image);  // Show our image inside it.

        if(cv::waitKey(30) >= 0)      //Escape enables to quit the program
            break;

    }
}

/** @function detectAndCut */
void detectAndCut(cv::Mat frame)
{
  std::vector<cv::Rect> faces;
  cv::Mat frame_gray;

  cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, cv::Size(30, 30) );

  if(faces.size() == 1) // Si qu'une face est détectée
  {
      cv::Mat faceROI = frame_gray( faces[0] );
      std::vector<cv::Rect> eyes;
      std::vector<cv::Point> eyescenters;

      //-- In face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0, cv::Size(30, 30) );

      if(eyes.size() == 2)
      {
          for( size_t j = 0; j < eyes.size(); j++ )
          {
             cv::Point center( faces[0].x + eyes[j].x + eyes[j].width*0.5, faces[0].y + eyes[j].y + eyes[j].height*0.5 );
             eyescenters.push_back(center);
             //std::cout << "Oeil " << j << " x=" << center.x << " y=" << center.y << std::endl;
          }
          if(eyescenters.at(1).x < eyescenters.at(0).x) std::reverse(eyescenters.begin(),eyescenters.end());

          std::string command = "python data/crop_face.py data/image.jpg ";
          command += std::to_string(eyescenters.at(0).x);
          command += " ";
          command += std::to_string(eyescenters.at(0).y);
          command += " ";
          command += std::to_string(eyescenters.at(1).x);
          command += " ";
          command += std::to_string(eyescenters.at(1).y);
          //std::cout<<"Commande exécutée : "<<command<<std::endl;
          int s = system(command.c_str());
          if(s != -1)
            ok = true;
      }
      else
      {
          std::cout<<"Erreur : yeux non détectés"<<std::endl;
      }
  }
  else
  {
      std::cout<<"Erreur : pas de face détectée"<<std::endl;
  }
}
