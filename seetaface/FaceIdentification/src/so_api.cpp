#include<iostream>
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif
extern "C"{
    float verify(char* img_path1, char* img_path2, char* det_model_path, char* alg_model_path, char* vrf_model_path, 
        int min_face_size, float score_thresh, float image_pyramid_scale_factor, int window_step_x, int window_step_y) {
	
      // Initialize face detection model
      seeta::FaceDetection detector(det_model_path);
      detector.SetMinFaceSize(min_face_size);
      detector.SetScoreThresh(score_thresh);
      detector.SetImagePyramidScaleFactor(image_pyramid_scale_factor);
      detector.SetWindowStep(window_step_x, window_step_y);
      // Initialize face alignment model 
      seeta::FaceAlignment point_detector(alg_model_path);

      // Initialize face Identification model 
      FaceIdentification face_recognizer(vrf_model_path);

      //load image
      cv::Mat gallery_img_color = cv::imread(img_path1, 1);
      cv::Mat gallery_img_gray;
      cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

      cv::Mat probe_img_color = cv::imread(img_path2, 1);
      cv::Mat probe_img_gray;
      cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

      ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
      gallery_img_data_color.data = gallery_img_color.data;

      ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
      gallery_img_data_gray.data = gallery_img_gray.data;

      ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
      probe_img_data_color.data = probe_img_color.data;

      ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
      probe_img_data_gray.data = probe_img_gray.data;

      // Detect faces
      std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
      int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

      std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
      int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

      if (gallery_face_num == 0 || probe_face_num==0)
      {
        //Faces are not detected
        return -1;
      }

      // Detect 5 facial landmarks
      seeta::FacialLandmark gallery_points[5];
      point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

      seeta::FacialLandmark probe_points[5];
      point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);

      // Extract face identity feature
      float gallery_fea[2048];
      float probe_fea[2048];
      face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
      face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

      // Caculate similarity of two faces
      float sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);

      return sim;
    }
}

