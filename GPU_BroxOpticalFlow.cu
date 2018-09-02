//executable program GetBroxOF, calculates Brox Flow Image from 2 input images using OpenCV with Cuda running on GPU
//Don Novkov, September 2018

//To use this program:
//1. Put this file (GPU_BroxOpticalFlow.cu), GPU_BroxOpticalFlow.h and CreateFlowImagesBrox.cpp into same folder.
//2. Change 'string ExtractedFramesLocation' and 'string SavedFlowImagesLocation' in this file to your addresses for those folders.
//3. Do the following compile (this code was run on 64-bit Ubuntu 14.04, Intel i5 CPU, NVIDIA Titan Xp GPU):

//g++ -c -I. CreateFlowImagesBrox.cpp -o CreateFlowImagesBrox.cpp.o

//nvcc -c -Wno-deprecated-gpu-targets -I. -I/usr/local/cuda-8.0/include -I/usr/local/include/opencv -I/usr/local/include/opencv2  GPU_BroxOpticalFlow.cu -o GPU_BroxOpticalFlow.cu.o

//g++ -o GetBroxOF GPU_BroxOpticalFlow.cu.o CreateFlowImagesBrox.cpp.o -L/usr/local/lib -L/usr/local/cuda-8.0/lib64 -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_cudaoptflow

//then run the executable from terminal:  ./GetBroxOF


#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <iostream>
#include <cuda.h>
#include <GPU_BroxOpticalFlow.h>

#include <cv.h> 
#include <highgui.hpp>
#include <opencv.hpp>
#include <cudaoptflow.hpp>

using namespace cv;
using namespace std;

void GpuInterface::GetFlowImage()
{
	Mat bgr;  //CV_32FC3 matrix
	DIR           *d;  //directory of Extracted Frames folders
	DIR	      *e;  //Saved Flow Files folder (created)
	DIR	      *f;  //Extracted Frames folder
	struct dirent *dir;
	struct dirent *dirf;	
	vector<string> dirlist;  //extracted frames folders
	string ExtractedFramesLocation = "/home/don/Documents/CNN-LSTM/UCF101/ExtractedFrames";
	string SavedFlowImagesLocation = "/home/don/Documents/CNN-LSTM/GetOpticalFlowBrox/SavedFlowImages";	
	d = opendir(ExtractedFramesLocation.c_str());
	
	Mat image1;
	int NumExtFramesFolders = 0;
	while ((dir = readdir(d)) != NULL)
	{
	    	string ReadDirectoryName = dir->d_name;	
		if(ReadDirectoryName.substr(0,1) != ".") {   //read all the relevant directories
			dirlist.push_back(dir->d_name);  //add this string to the dirlist vector
			NumExtFramesFolders++;
		}
	}
	closedir(d);  //all relevant directories have been read
	sort( dirlist.begin(), dirlist.end() );  //sort the filtered directory list alphabetically
	for(int i=0; i<NumExtFramesFolders; i++) {  //cycle thru all extracted frames folders and calc optical flows
		string CheckFolder = SavedFlowImagesLocation + "/" + dirlist[i];
		e = opendir(CheckFolder.c_str());
		if (e) {closedir(e);}  //directory exists, don't overwrite it
		else { //directory doesn't exist, so make the optical flow files
			mkdir(CheckFolder.c_str(), 0700);  //make directory
			string ReadFolder = ExtractedFramesLocation + "/" + dirlist[i];
			vector<string> dirlistf; //individual frame files within extracted frames folder
			f = opendir(ReadFolder.c_str());
			printf("Making Optical Flow Files: %s\n",dirlist[i].c_str());
			int NumOfFrames = 0;
			while ((dirf = readdir(f)) != NULL)
			{
				string FrameFileName = dirf->d_name;				
				if(FrameFileName.substr(0,1) != ".") {
					dirlistf.push_back(dirf->d_name);  //add this string to the dirlistf vector
					NumOfFrames++;
				}
			}
			closedir(f);  //all relevant directories have been read
			sort( dirlistf.begin(), dirlistf.end() );  //sort the filtered directory list alphabetically
			string Image0FileName = ReadFolder + "/" + dirlistf[0];  //read first frame file
			Mat image0 = imread(Image0FileName.c_str() , 0);
			for(int j=1; j<NumOfFrames; j++) {  //cycle thru all frames in folder		
			    	string Image1FileName = ReadFolder + "/" + dirlistf[j];	
				//now run the code below within this for loop:
	
	//tips from these two websites were used for the cuda code:
	//https://stackoverflow.com/questions/15069255/opencv-brox-optical./exec-flow-exception-in-opencv-core244dcvglbufferunbind
	//https://github.com/antran89/OpticalFlow-comparisons/blob/master/gpuOpticalFlow.cpp
	
	image1 = imread(Image1FileName.c_str() , 0);

    	Mat image0Float; // will contain images in format CV_32FC1
	Mat image1Float;

	image0.convertTo(image0Float, CV_32FC1, 1.0/255.0);
	image1.convertTo(image1Float, CV_32FC1, 1.0/255.0);

	// Upload images to GPU
    	cuda::GpuMat image0GPU(image0Float);
	cuda::GpuMat image1GPU(image1Float);
	//gpu module was redesigned in OpenCV 3.0. It was split; renamed to cuda and gpu:: namespace renamed to cuda::

	// Prepare receiving variable
	cuda::GpuMat FlowGPU;
	
	// Create optical flow object: https://docs.opencv.org/3.3.0/d7/d18/classcv_1_1cuda_1_1BroxOpticalFlow.html
	//(double alpha=0.197, double gamma=50.0, double scale_factor=0.8, int inner_iterations=5,
	// int outer_iterations=150, int solver_iterations=10)
	Ptr<cuda::BroxOpticalFlow> broxFlow = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 5, 150, 10);	
	broxFlow->calc(image0GPU, image1GPU, FlowGPU);  //input images need to be CV_32FC1
	Mat flow;
	FlowGPU.download(flow);  //returns flow as CV_32FC2 (2 channels, [i,j,n] where n ranges from 0 to 1)

	Mat xy[2]; 
	split(flow, xy); //split flow into two 1-channel images, X and Y

	//following is Lisa Anne Hendrick's tutorial https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video .m file routine recoded in OpenCV:	
	int scale = 16;
	Mat mag, dst0, dst1, dstZ;
	//mag = sqrt(flow(:,:,1).^2+flow(:,:,2).^2)*scale+128 is split into following 4 equations:
	pow(xy[0], 2, dst0);	//1
	pow(xy[1], 2, dst1);	//2
	sqrt(dst0+dst1 , dstZ);	//3
        mag = dstZ*scale+128;	//4
	mag = min(mag, 255);
	flow = flow*scale+128;
        flow = min(flow,255);
        flow = max(flow,0);
	
	Mat _FlowImage[3], FlowImage;	//create an HSV image from 2 channels plus magnitude
	_FlowImage[0] = xy[0];
	_FlowImage[1] = xy[1];
	_FlowImage[2] = mag;
	merge(_FlowImage, 3, FlowImage);

	cvtColor(FlowImage, bgr, COLOR_HSV2BGR);  //convert to BGR
	//cvtColor(FlowImage, bgr, COLOR_HSV2RGB);  //convert to RGB
	
	string SaveToFileName = CheckFolder + "/flow_image_" + dirlistf[j];
	imwrite( SaveToFileName.c_str() , bgr );
	image0 = image1;

			}  //for
			closedir(e);
		}  //else
	}  //for

	return;
}

