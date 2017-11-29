//
//  test_max_sift.hpp
//  MyfirstOpenCV
//
//  Created by Xu Muxin on 2017/11/19.
//  Copyright © 2017年 Xu Muxin. All rights reserved.
//

#ifndef test_max_sift_hpp
#define test_max_sift_hpp

#include <iostream>
#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "max_sift.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "max_sift.hpp"
using namespace cv;  //
using namespace std;

class TestMaxSIFT {
public:
    MaxSIFT MAXSIFT;
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
public:
    int QUERT_SIZE = 4; // 基准图片的数量
    // 获取某个文件夹下所有图片的路径，存储在Vector里
    TestMaxSIFT(int Q) {
        QUERT_SIZE = Q;
    }
    
    ~TestMaxSIFT() {}
    
    // test
    vector<string> getFiles(string cate_dir)
    {
        vector<string> files;//存放文件名
        
        DIR *dir;
        struct dirent *ptr;
        char base[1000];
        string p;
        
        if ((dir=opendir(cate_dir.c_str())) == NULL)
        {
            cout << "Open dir error..." << endl;
            
        }
        
        while ((ptr=readdir(dir)) != NULL)
        {
            if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
                continue;
            else if(ptr->d_type == 8)    ///file
                //printf("d_name:%s/%s\n",basePath,ptr->d_name);
                files.push_back(ptr->d_name);
            //cout << ptr->d_name << endl;
            
            else if(ptr->d_type == 10)    ///link file
                //printf("d_name:%s/%s\n",basePath,ptr->d_name);
                continue;
            else if(ptr->d_type == 4)    ///dir
            {
                //files.push_back(p.assign(cate_dir).append("/").append(ptr->d_name));
                files.push_back(ptr->d_name);
                /*
                 memset(base,'\0',sizeof(base));
                 strcpy(base,basePath);
                 strcat(base,"/");
                 strcat(base,ptr->d_nSame);
                 readFileList(base);
                 */
            }
        }
        closedir(dir);
        
        //排序，按从小到大排序
        sort(files.begin(), files.end());
        return files;
    }
    
    // 获取每个品牌图片的图片路径，分别放在不同的Vector中
    vector<vector<string> > getAllFiles(string cate_dir) {
        vector<string> subdir = getFiles(cate_dir);
        cout << cate_dir << endl;
        
        cout << subdir.size() <<endl;
        vector<vector<string> > total(51, vector<string>());
        string p;
        vector<string> temp;
        for (int i = 0; i < 51; ++i) {
            temp = getFiles(p.assign(cate_dir).append("/").append(subdir[i]));
            for (int j = 0; j < temp.size(); ++j) {
                temp[j] = p.assign(cate_dir).append("/").append(subdir[i]).append("/").append(temp[j]);
            }
            total[i] = temp;
            // cout << i << " " << p.assign(cate_dir).append("/").append(subdir[i]) << endl;
        }
        return total;
    }
    
    // test
    // 获取所有基准图片的关键点
    vector<Mat> getBaseDes(vector<vector<string>> all) {
        vector<Mat> total(51 * QUERT_SIZE, Mat());
        for (int i = 0; i < 51; ++i) {
            for (int j = 0; j < QUERT_SIZE; ++j) {
                cout << i * QUERT_SIZE + j << endl;
                Mat baseImg = imread(all[i][j], IMREAD_GRAYSCALE);
                vector<KeyPoint> keypoints;
                f2d->detect(baseImg, keypoints);
                f2d->compute(baseImg, keypoints, total[i * QUERT_SIZE + j]);
            }
        }
        return total;
    }
    
    // 用SIFT方法来匹配这些图片与基准图片，选取匹配点数最多的品牌作为结果
    int SiftCompareWithAllBasePicture(Mat img_test, vector<Mat> totalDescriptors) {
        int pos = 0, max = -1;
        vector<KeyPoint> keypoints_test;
        Mat descriptors_test;
        f2d->detect(img_test, keypoints_test);
        f2d->compute(img_test, keypoints_test, descriptors_test);
        std::vector<DMatch> matches;
        for (int i = 0; i < 51; ++i) {
            int size = 0;
            for (int j = 0; j < QUERT_SIZE; ++j) {
                MAXSIFT.ratioTest(matches, descriptors_test, totalDescriptors[i * QUERT_SIZE + j]);
                size += matches.size();
            }
            if (size > max) {
                pos = i;
                max = size;
            }
        }
        return pos;
    }
    
    void showTheMatch(Mat img_test, vector<KeyPoint> keypoints_test, Mat img, vector<KeyPoint> keypoints, vector<DMatch> matches) {
        Mat img_matches;
        drawMatches(img_test,keypoints_test,img,keypoints,matches,img_matches,
                    Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);
        imshow("max-sift", img_matches);
        waitKey(0);
    }

    // 展示一张图片与查询图片的query_test
    void showOneImage(string test, string query) {
        Mat image_test = imread(test, IMREAD_GRAYSCALE);
        Mat image_query = imread(query, IMREAD_GRAYSCALE);
        Mat descriptors_test, descriptors_query;
        vector<KeyPoint> keypoints_test, keypoints_query;
        f2d->detect(image_test, keypoints_test);
        f2d->compute(image_test, keypoints_test, descriptors_test);
        f2d->detect(image_query, keypoints_query);
        f2d->compute(image_query, keypoints_query, descriptors_query);
        vector<DMatch> matches;
        MAXSIFT.ratioTest(matches, descriptors_test, descriptors_query);
        // SIFT 图片
        showTheMatch(image_test, keypoints_test, image_query, keypoints_query, matches);
        
        

        float* test;
        for (int i = 0; i < descriptors_test.rows; i++) {
            test = descriptors_test.ptr<float>(i);
            MAXSIFT.max_sift(test);
        }
        for (int i = 0; i < descriptors_query.rows; i++) {
            test = descriptors_query.ptr<float>(i);
            MAXSIFT.max_sift(test);
        }
        MAXSIFT.ratioTest(matches, descriptors_test, descriptors_query);
        showTheMatch(image_test, keypoints_test, image_query, keypoints_query, matches);



    }
    
    // 用MAX-SIFT方法来匹配这些图片与基准图片，选取匹配点数最多的品牌作为结果，并可以显示出两张图片的匹配结果
    // 之前用于检查错误
    int MaxsiftComCompareWithAllBasePicture(Mat img_test, vector<Mat> totalDescriptors, vector<vector<string> > all, bool show) {
        int pos = 0, max = -1;
        vector<KeyPoint> keypoints_test, keypoints;
        Mat descriptors_test;
        f2d->detect(img_test, keypoints_test);
        f2d->compute(img_test, keypoints_test, descriptors_test);
        float* test;
        for (int i = 0; i < descriptors_test.rows; i++) {
            test = descriptors_test.ptr<float>(i);
            MAXSIFT.max_sift(test);
        }
        for (int i = 0; i < 51; ++i) { // 对51张基准图片采用max_sift方法
            
            for (int j = 0; j < totalDescriptors[i].rows; j++) {
                test = totalDescriptors[i].ptr<float>(j);
                MAXSIFT.max_sift(test);
            }
        }
        std::vector<DMatch> matches;
        for (int i = 0; i < 51; ++i) {
            Mat img = imread(all[i][0], IMREAD_GRAYSCALE);
            f2d->detect(img, keypoints);
            Mat temp;
            f2d->compute(img, keypoints, temp);
            MAXSIFT.ratioTest(matches, descriptors_test, totalDescriptors[i]);
            if (show) {
                showTheMatch(img_test, keypoints_test, img, keypoints, matches);
            }
            int size = 0;
            for (int j = 0; j < QUERT_SIZE; ++j) {
                MAXSIFT.ratioTest(matches, descriptors_test, totalDescriptors[i * QUERT_SIZE + j]);
                size += matches.size();
            }
            
            if (size > max) {
                pos = i;
                max = size;
            }
            
        }
        return pos;
    }
    

    
    
    // 用MAX-SIFT方法来匹配这些图片与基准图片，选取匹配点数最多的品牌作为结果
    int MaxsiftComCompareWithAllBasePicture(Mat img_test, vector<Mat> totalDescriptors) {
        int pos = 0, max = -1;
        vector<KeyPoint> keypoints_test, keypoints;
        Mat descriptors_test;
        f2d->detect(img_test, keypoints_test);
        f2d->compute(img_test, keypoints_test, descriptors_test);
        float* test;
        for (int i = 0; i < descriptors_test.rows; i++) {
            test = descriptors_test.ptr<float>(i);
            MAXSIFT.max_sift(test);
        }
        for (int i = 0; i < 51; ++i) { // 对51张基准图片采用max_sift方法
            
            for (int j = 0; j < totalDescriptors[i].rows; j++) {
                test = totalDescriptors[i].ptr<float>(j);
                MAXSIFT.max_sift(test);
            }
        }
        std::vector<DMatch> matches;
        for (int i = 0; i < 51; ++i) {
            //Mat img = imread(all[i][0]);
            //f2d->detect(img, keypoints);
            //Mat temp;
            //f2d->compute(img, keypoints, temp);
            
            //if (show) {
            //showTheMatch(img_test, keypoints_test, img, keypoints, matches);
            //}
            int size = 0;
            for (int j = 0; j < QUERT_SIZE; ++j) {
                MAXSIFT.ratioTest(matches, descriptors_test, totalDescriptors[i * QUERT_SIZE + j]);
                size += matches.size();
            }
            
            
            if (size > max) {
                pos = i;
                max = size;
            }
            
        }
        return pos;
    }
    
    // test over
};




#endif /* test_max_sift_hpp */
