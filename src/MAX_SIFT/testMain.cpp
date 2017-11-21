#include "test_max_sift.hpp"


int main()
{
	int number = 0;
	cin >> number;
    TestMaxSIFT test_maxsift(number);
    string path = "/root/Downloads/CarLogos51";
    
	vector<float> re;
    vector<vector<string> > all = test_maxsift.getAllFiles(path);

    vector<Mat> totalDescriptors = test_maxsift.getBaseDes(all);
	int numberOfRightBySift = 0, eachLogoBySift = 0;
    int numberOfRightByMax = 0, eachLogoByMax = 0;
	int sum = 0;
	ofstream fout;


	fout.open("result24v1");
	cout << all[0][0] << endl;
	for (int i = 0; i < 51; ++i) {
        eachLogoByMax = 0;
        eachLogoBySift = 0;
        // if (i == 0) continue;
        cout << i << endl;
        // fout << i << endl;
        for (int j = test_maxsift.QUERT_SIZE; j < all[i].size(); ++j) {
            cout << "\t" << j << endl;
            int pos1, pos2;
            Mat img_test = imread(all[i][j]);
            pos1 = test_maxsift.SiftCompareWithAllBasePicture(img_test, totalDescriptors);
			pos2 = test_maxsift.MaxsiftComCompareWithAllBasePicture(img_test, totalDescriptors);
			
            if (pos1 == i) {
                ++numberOfRightBySift;
                ++eachLogoBySift;
            }
            if (pos2 == i) {
                ++numberOfRightByMax;
                ++eachLogoByMax;
            }
        }
        float r1 = eachLogoBySift / (float) all[i].size();
        float r2 = eachLogoByMax / (float) all[i].size();
		sum += all[i].size();
        fout << i << " SIFT  " << r1 << "  " << numberOfRightBySift << endl;
		fout << i << " MAX-SIFT  " << r2 << "  " << numberOfRightByMax << endl;
		
		
	}
	
	fout << "SIFT Final Accuracy " << numberOfRightBySift / (float) sum << endl;
	fout << "MAX-SIFT Final Accuracy " << numberOfRightByMax / (float) sum << endl;
	fout << flush;
	fout.close();
    
 	
    waitKey(0);
}
