#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define VAN_PNG_PATH "data/1P.png"
#define LEO_PNG_PATH "data/0P.png"
#define VAN_MARK_PATH "data/1M.png"
#define LEO_MARK_PATH "data/0M.png"

cv::Scalar RED = cv::Scalar(0, 0, 255);
cv::Scalar BLUE = cv::Scalar(255, 0, 0);

// Rotate the vector: removes the last element and put it in front 
std::vector<cv::Point2f> getRotatedVector( std::vector<cv::Point2f> vec)
{
    std::vector<cv::Point2f> aux; 

    cv::Point2f last = vec.back();
    vec.pop_back();
    aux.push_back(last);

    for( size_t i=0; i < vec.size(); ++i)
    {
        aux.push_back(vec[i]);
    }

    return aux;
}

// Main program code
int main( int argc, char* argv[] )
{
    // Tunables
    int MIN_CONTOUR_SIZE = 100,
        GAUSS_SIGMA_X = 2,
        GAUSS_SIGMA_Y = 2,
        GAUSS_SIZE = 5;

    float EPSILON_APPROX = 6.0,
          GAIN = 1.5,
          BIAS = 0.0,
          MIN_CONTOUR_AREA = 1000.0,
          MIN_LEO_MATCHING_SCORE = 0.88 ,
          MIN_VAN_MATCHING_SCORE = 0.88 ;
        
    
    // Loading Markers and Isolation and blurring of L of Leo and V of Van 
    cv::Rect first_letter_roi = cv::Rect(65, 150, 60, 54);
    cv::Mat van_marker = cv::imread(VAN_MARK_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat leo_marker = cv::imread(LEO_MARK_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat L = leo_marker(first_letter_roi);
    cv::Mat V = van_marker(first_letter_roi);
    cv::GaussianBlur(L,L, cv::Size(GAUSS_SIZE,GAUSS_SIZE),GAUSS_SIGMA_X,GAUSS_SIGMA_Y);
    cv::GaussianBlur(V,V, cv::Size(GAUSS_SIZE,GAUSS_SIZE),GAUSS_SIGMA_X,GAUSS_SIGMA_Y);

    // Loading paintings 
    cv::Mat van_painting = cv::imread(VAN_PNG_PATH);
    cv::Mat leo_painting = cv::imread(LEO_PNG_PATH);

    // Marker mask for final warp
    cv::Mat small_marker_mask = cv::Mat(van_marker.rows, van_marker.cols, CV_8U, cv::Scalar(255));

    // Marker 4 reference points
    std::vector<cv::Point2f> mark_ref_points;
    mark_ref_points.push_back(cv::Point2f(0,0));
    mark_ref_points.push_back(cv::Point2f(leo_painting.cols-1,0));
    mark_ref_points.push_back(cv::Point2f(leo_painting.cols-1,leo_painting.rows-1));
    mark_ref_points.push_back(cv::Point2f(0,leo_painting.rows-1));

    // Aux variables
    cv::Mat frame, orig_frame, contour_frame,warped_painting, warped_mask, h, detected_marker, detected_L, detected_V ;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Point2f>  approx_cont, detected_corners;
    std::vector<float> leo_score, van_score;

    // Start video source
    cv::VideoCapture vc;

    // If you don't have a webcam just put the path to a valid video which shows the markers
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

                // Gain filter
                frame.convertTo(frame,-1,GAIN,BIAS); 
                // [DEBUG]
                //cv::imshow("GAIN filter", frame);
                
                // Threshold
                cv::threshold( frame, frame, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
                
                // Find Contours
                cv::findContours(frame, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

                // Detected contours
                for (size_t idx = 0; idx < contours.size(); ++idx)
                {
                    // Filtering contours by pixel number
                    if (contours[idx].size() > MIN_CONTOUR_SIZE)
                    {
                        // Contour shape approximation through RDP-algorithm
                        cv::approxPolyDP( contours[idx], approx_cont, EPSILON_APPROX ,true);

                        // Square shape filtering
                        if( approx_cont.size() == 4 && cv::isContourConvex(approx_cont) && fabs(cv::contourArea(approx_cont)) > MIN_CONTOUR_AREA )
                        {
                            // [DEBUG] We are sure it is a square so we draw the detected contour
                            cv::drawContours(contour_frame, contours, idx, RED);

                            // [DEBUG] Draw a circle over the corners
                            cv::circle(contour_frame, approx_cont[0], 10.0, BLUE);
                            cv::circle(contour_frame, approx_cont[1], 10.0, BLUE);
                            cv::circle(contour_frame, approx_cont[2], 10.0, BLUE);
                            cv::circle(contour_frame, approx_cont[3], 10.0, BLUE);

                            // [DEBUG]
                            //cv::imshow("Detected Squares", contour_frame);

                            // Square corners
                            detected_corners.clear();

                            // Save corner points of the detected square
                            detected_corners.push_back(approx_cont[0]);
                            detected_corners.push_back(approx_cont[1]);
                            detected_corners.push_back(approx_cont[2]);
                            detected_corners.push_back(approx_cont[3]);

                            // For each possible orientation
                            for( int orientation = 0; orientation <4; ++orientation)
                            {
                                // Marker matching 
                                h = cv::getPerspectiveTransform(detected_corners, mark_ref_points);
                                cv::warpPerspective(frame, detected_marker, h, warped_painting.size());

                                // Filter only the first letter of the marker 
                                detected_L = detected_marker(first_letter_roi);
                                detected_V = detected_marker(first_letter_roi);
                                // [DEBUG]
                                //cv::imshow("Detected Marker", detected_marker);
                                
                                // Template matching
                                cv::matchTemplate(detected_L, L, leo_score, cv::TM_CCORR_NORMED);
                                cv::matchTemplate(detected_V, V, van_score, cv::TM_CCORR_NORMED);

                                // If it matches one of the two markers, then display the correct one
                                if( van_score.back() > MIN_VAN_MATCHING_SCORE ||  leo_score.back() > MIN_LEO_MATCHING_SCORE )
                                { 
                                    // Calculate the transformation matrix
                                    h = cv::getPerspectiveTransform(mark_ref_points, detected_corners);

                                    warped_painting = cv::Mat(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0,0,0));
                                    warped_mask = cv::Mat(frame.rows, frame.cols, CV_8U, cv::Scalar(255));

                                    // L or V?
                                    if ( leo_score.back() > van_score.back())
                                    {
                                        // [DEBUG]
                                        //cv::imshow("L", detected_L);
                                        cv::warpPerspective(leo_painting, warped_painting, h, warped_painting.size());
                                    }
                                    else
                                    {
                                        // [DEBUG]
                                        //cv::imshow("V", detected_V);
                                        cv::warpPerspective(van_painting, warped_painting, h, warped_painting.size());
                                    }

                                    // Warp the mask an copy the warped image
                                    cv::warpPerspective(small_marker_mask, warped_mask, h, warped_mask.size());
                                    warped_painting.copyTo(orig_frame, warped_mask);

                                    // Stop trying every orientation and process another square
                                    break;
                                }

                                // Try another orientation
                                detected_corners = getRotatedVector(detected_corners);
                            }
                        }
                    }
                }

                cv::imshow( "Augmented Reality", orig_frame);

                // [DEBUG]
                //cv::imshow( "Binary image", frame);
                //cv::imshow("Detected Squares", contour_frame);

                cv::waitKey( 1 );
            }
            else
            {
                // No more frames, exit.
                break;
            }
        }
    
    return 0;

    }
}

