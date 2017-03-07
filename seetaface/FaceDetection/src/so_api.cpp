#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"
using namespace std;
struct Face{
	bool null = true;
	int x, y, width, height;
	Face* next = NULL;
};
extern "C"
{
    Face* detect(char* img_path, char* model_path, int min_face_size, float score_thresh,
		float image_pyramid_scale_factor, int window_step_x, int window_step_y){
        seeta::FaceDetection detector(model_path);
	detector.SetMinFaceSize(min_face_size);
        detector.SetScoreThresh(score_thresh);
        detector.SetImagePyramidScaleFactor(image_pyramid_scale_factor);
        detector.SetWindowStep(window_step_x, window_step_y);
	
        cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
        cv::Mat img_gray;
        if (img.channels() != 1)
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        else
            img_gray = img;
	
        seeta::ImageData img_data;
        img_data.data = img_gray.data;
        img_data.width = img_gray.cols;
        img_data.height = img_gray.rows;
        img_data.num_channels = 1;
        std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
	Face* root = new Face;
        Face* now = root;
	
        for(int i = 0; i < faces.size(); i++){
            now->width = faces[i].bbox.width;
            now->height = faces[i].bbox.height;
            now->x = faces[i].bbox.x;
            now->y = faces[i].bbox.y;
            now->null = false;
            now->next = new Face;
            now = now->next; 
        }
	
        return root;
    }
}

