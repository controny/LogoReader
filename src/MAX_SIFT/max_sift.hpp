//
//  max_sift.hpp
//  MyfirstOpenCV
//
//  Created by Xu Muxin on 2017/11/19.
//  Copyright © 2017年 Xu Muxin. All rights reserved.
//

#ifndef max_sift_hpp
#define max_sift_hpp

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
using namespace cv;  //
using namespace std;

class MaxSIFT {
private:
    // the sequence of smallest indexes in the 32 groups
    int L[32] ={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
               ,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47};
    // the other 3 inndex sequences of 3 different flipping operations
    int G[32],B[32],GB[32];
    typedef int (*pfun)(int i);
public:
    MaxSIFT() {
        init();
    }
    ~MaxSIFT() {}
    
    // map the sequence L to the other 3 sequences according to diffrent flipping operations
    void init() {
        for (int i = 0; i < 32; i++) {
            G[i] = geometry_inverted(L[i]);
        }
        for (int i = 0; i < 32; i++) {
            B[i] = brightness_inverted(L[i]);
        }
        for (int i = 0; i < 32; i++) {
            GB[i]=brightness_inverted(geometry_inverted(L[i]));
        }
    }
    // map the index in the L to the index in the G(geometry_ivt)
    static int geometry_inverted(int i) {
        int bin = i/8;
        int d = i%8;
        bin = (bin/4)*4+3-bin%4;
        d = (8-d)%8;
        return bin*8+d;
    }
    // map the index in the L to the index in the B(brightness_ivt)
    static int brightness_inverted(int i) {
        int bin = i/8;
        int d = i%8;
        bin = 15-bin;
        return bin*8+d;
    }
    // map the index in the L to the index in the GB(geometry_brightness_ivt)
    static int geo_bright_inverted(int i) {
        return brightness_inverted(geometry_inverted(i));
    }
    
    // generate the value seq according to the indexes in each index seq
    void getValueSequence(float* V, float* VG, float* VB, float* VGB, float* p) {
        for (int s = 0; s < 32; s++) {
            V[s] = *(p+L[s]);
            VG[s] = *(p+G[s]);
            VB[s] = *(p+B[s]);
            VGB[s] = *(p+GB[s]);
        }
    }
    // fing the value seq with the maximal alphabetical order
    int findTheDominantValueSeq(vector<float*> all) {
        int win[4] = {0,0,0,0};
        for (int i = 0; i < 32; i++) {
            int sflag = 0;
            for (int j = 0; j < 4; j++) {
                sflag += win[j];
            }
            if (sflag == -3) break;
            int t = 0;
            while (win[t] == -1) t++;
            for (int index = 1; index <= 3; index++) {
                int another =(t+index)%4;
                if (win[another] != -1) {
                    if (all[t][i] < all[another][i]) {
                        win[t] = -1;
                        t = another;
                        index = 0;
                    }
                    else if (all[t][i] > all[another][i]) {
                        win[another] = -1;
                    }
                }
            }
        }
        int which = 0;
        while (win[which] != 0) which++;
        return which;
    }
    
    void revertTheDescriptor(float* p,pfun f) {
        vector<float> temp(p, p + 128);
        for (int i = 0; i < 128; i++) {
            int where = f(i);
            p[i] = temp[where];
        }
    }
    
    void max_sift(float* p) {
        // generate 4 value sequences according to 4 index sequences
        float V[32],VG[32],VB[32],VGB[32];
        getValueSequence(V, VG, VB, VGB, p);
        
        // find the dominant value sequence
        vector<float*> all = {V,VG,VB,VGB};
        int which = findTheDominantValueSeq(all);
       
        // revert the descriptor with the operationn represented by the dominant value sequence
        switch (which) {
            case 1:
                revertTheDescriptor(p, geometry_inverted);
                break;
            case 2:
                revertTheDescriptor(p, brightness_inverted);
                break;
            case 3:
                revertTheDescriptor(p, geo_bright_inverted);
                break;
            default:
                break;
        }
    }
    
    void ratioTest(std::vector<DMatch>& output,Mat query_test, Mat train) {
        output.clear();
        BFMatcher matcher;
        std::vector<std::vector<DMatch> > matches2;
        matcher.knnMatch(query_test, train, matches2, 2);
        const float minRatio = 0.85;
        for (int i=0; i<matches2.size(); i++)
        {
            const cv::DMatch& bestMatch = matches2[i][0];
            const cv::DMatch& betterMatch = matches2[i][1];
            float distanceRatio = bestMatch.distance /betterMatch.distance;
            if (distanceRatio < minRatio)
            {
                output.push_back(bestMatch);
            }
        }
    }
    
};

#endif /* max_sift_hpp */
