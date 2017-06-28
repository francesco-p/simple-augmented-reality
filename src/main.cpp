#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define VAN_PNG_PATH "data/1P.png"
#define LEO_PNG_PATH "data/0P.png"
#define VAN_MARK_PATH "data/1M.png"
#define LEO_MARK_PATH "data/0M.png"

int TOT_SCORE = 0;
cv::Scalar RED = cv::Scalar(0, 0, 255);
cv::Scalar BLUE = cv::Scalar(255, 0, 0);

// Shift by one position and rotate the deque
void dequeRotate(std::deque<cv::Point2f> &dq)
{
    cv::Point2f aux = dq.back();     
    dq.pop_back();
    dq.push_front(aux);
}

void vectorToDeque( std::vector<cv::Point2f> vec, std::deque<cv::Point2f> &dq)
{   
    dq.clear();

    for( size_t i=0; i < vec.size(); ++i)
    {
        dq.push_back(vec[i]);
    }
}

void dequeToVector( std::deque<cv::Point2f> dq, std::vector<cv::Point2f> &vec)
{   
    vec.clear();

    for( size_t i=0; i < dq.size(); ++i)
    {
        vec.push_back(dq[i]);
    }
}


int main( int argc, char* argv[] )
{
    // Tunables
    int MIN_CONTOUR_SIZE = 100,
        BIN_THRESH = 57;

    float EPSILON_APPROX = 6.0,
          MIN_CONTOUR_AREA = 1000.0,
          MIN_LEO_MATCHING_SCORE = 0.88 ,
          MIN_VAN_MATCHING_SCORE = 0.88 ;
        
    
    // Loading Markers and Isolation and blurring of L of Leo and V of Van 
    cv::Mat van_marker = cv::imread(VAN_MARK_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat leo_marker = cv::imread(LEO_MARK_PATH, cv::IMREAD_GRAYSCALE);
    cv::Rect first_letter_roi = cv::Rect(65, 150, 60, 54);
    cv::Mat L = leo_marker(first_letter_roi);
    cv::Mat V = van_marker(first_letter_roi);
    cv::GaussianBlur(L,L, cv::Size(5,5),2,2);
    cv::GaussianBlur(V,V, cv::Size(5,5),2,2);
    // Marker mask for final warp
    cv::Mat small_marker_mask = cv::Mat(van_marker.rows, van_marker.cols, CV_8U, cv::Scalar(255));

    // Loading paintings 
    cv::Mat van_painting = cv::imread(VAN_PNG_PATH);
    cv::Mat leo_painting = cv::imread(LEO_PNG_PATH);

    // Marker reference points
    std::vector<cv::Point2f> mark_ref_points;
    mark_ref_points.push_back(cv::Point2f(0,0));
    mark_ref_points.push_back(cv::Point2f(leo_painting.cols-1,0));
    mark_ref_points.push_back(cv::Point2f(leo_painting.cols-1,leo_painting.rows-1));
    mark_ref_points.push_back(cv::Point2f(0,leo_painting.rows-1));

    // Aux variables
    cv::Mat frame, orig_frame, contour_frame,warped_painting, warped_mask, h, detected_marker, detected_L, detected_V ;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Point2f>  approx_cont, detected_corners_vec;
    std::deque<cv::Point2f> detected_corners_dq;
    std::vector<float> leo_score, van_score;

    // Start video source
    cv::VideoCapture vc;

    // If you don't have a webcam, just put the path of a videofile
    vc.open(0);

    if(vc.isOpened())
    {
        while(true)
        {
            vc >> orig_frame;
            frame = orig_frame.clone();
            contour_frame = cv::Mat(orig_frame.size(), CV_8UC3, cv::Scalar(0,0,0));

            if (!frame.empty())
            {

                // Greyscale
                cv::cvtColor( frame, frame, CV_RGB2GRAY );

                // 2xGain
                frame.convertTo(frame,-1,2,0); 
                
                // Threshold
                //cv::threshold( frame, frame, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
                cv::threshold( frame, frame, BIN_THRESH, 255, CV_THRESH_BINARY);
                
                // Image processing
                //cv::blur(frame,frame, cv::Size(3,3), cv::Point(-1,-1));
                //cv::GaussianBlur(frame,frame, cv::Size(3,3),2,2);

                // Find Contours
                cv::findContours(frame, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

                warped_painting = cv::Mat(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0,0,0));
                warped_mask = cv::Mat(frame.rows, frame.cols, CV_8U, cv::Scalar(255));

                // Detected contours
                for (size_t idx = 0; idx < contours.size(); ++idx)
                {

                    // Filtering by pixel number
                    if (contours[idx].size() > MIN_CONTOUR_SIZE)
                    {
                        // Shape approximation
                        cv::approxPolyDP( contours[idx], approx_cont, EPSILON_APPROX ,true);

                        // Square shape filtering
                        if( approx_cont.size() == 4 && cv::isContourConvex(approx_cont) && fabs(cv::contourArea(approx_cont)) > MIN_CONTOUR_AREA )
                        {
                            // [DEBUG] We are sure it is a square so we draw the detected contour
                            cv::drawContours(contour_frame, contours, idx, RED);

                            // Corners
                            for (int j = 0; j < approx_cont.size(); j+=4)
                            {
                                detected_corners_vec.clear();
                                detected_corners_dq.clear();

                                // [DEBUG] Draw corners
                                cv::circle(contour_frame, approx_cont[j], 10.0, BLUE);
                                cv::circle(contour_frame, approx_cont[j+1], 10.0, BLUE);
                                cv::circle(contour_frame, approx_cont[j+2], 10.0, BLUE);
                                cv::circle(contour_frame, approx_cont[j+3], 10.0, BLUE);

                                // Save corner points
                                detected_corners_vec.push_back(approx_cont[j]);
                                detected_corners_vec.push_back(approx_cont[j+1]);
                                detected_corners_vec.push_back(approx_cont[j+2]);
                                detected_corners_vec.push_back(approx_cont[j+3]);

                                vectorToDeque(detected_corners_vec, detected_corners_dq);

                                // For each orientation
                                for( int orientation = 0; orientation <4; ++orientation)
                                {
                                    // Marker matching 
                                    h = cv::getPerspectiveTransform(detected_corners_vec, mark_ref_points);
                                    cv::warpPerspective(frame, detected_marker, h, warped_painting.size());

                                    // Which marker
                                    detected_L = detected_marker(first_letter_roi);
                                    detected_V = detected_marker(first_letter_roi);
                                    // [DEBUG]
                                    //cv::imshow("Detected Marker", detected_marker);
                                    
                                    // Template matching
                                    cv::matchTemplate(detected_L, L, leo_score, cv::TM_CCORR_NORMED);
                                    //std::cout <<  "LEO " << leo_score.back() << std::endl;
                                    cv::matchTemplate(detected_V, V, van_score, cv::TM_CCORR_NORMED);
                                    //std::cout <<  "VAN " << van_score.back() << std::endl;

                                    h = cv::getPerspectiveTransform(mark_ref_points, detected_corners_vec);
                                    //h = cv::findHomography(mark_ref_points, detected_corners_vec, cv::RANSAC);

                                    if( van_score.back() > MIN_VAN_MATCHING_SCORE ||  leo_score.back() > MIN_LEO_MATCHING_SCORE )
                                    { 
                                        if ( leo_score.back() > van_score.back())
                                        {
                                            // [DEBUG]
                                            cv::imshow("L", detected_L);

                                            cv::warpPerspective(leo_painting, warped_painting, h, warped_painting.size());
                                        }
                                        else
                                        {
                                            // [DEBUG]
                                            cv::imshow("V", detected_V);

                                            cv::warpPerspective(van_painting, warped_painting, h, warped_painting.size());
                                        }

                                        cv::warpPerspective(small_marker_mask, warped_mask, h, warped_mask.size());
                                        warped_painting.copyTo(orig_frame, warped_mask);

                                        // [DEBUG]
                                        //TOT_SCORE++;

                                        break;
                                    }

                                    dequeRotate(detected_corners_dq);
                                    dequeToVector(detected_corners_dq, detected_corners_vec);
                                    //cv::waitKey(0);
                                }
                            }
                        }
                    }
                }

                cv::imshow( "Augmented Reality", orig_frame);

                // [DEBUG]
                cv::imshow( "Binary image", frame);
                cv::imshow("Detected Squares", contour_frame);

                cv::waitKey( 1 );
            }
            else
            {
                break;
            }
        }
    
    // [DEBUG]
    //std::cout <<  "Tot score: " << TOT_SCORE << std::endl;

    return 0;

    }
}

