#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <sstream>
#include <profile.h>

using namespace std;
using namespace cv;

// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

// create tracker by name
Ptr<Tracker> createTrackerByName(string trackerType)
{
  Ptr<Tracker> tracker;
  if (trackerType ==  trackerTypes[0])
    tracker = TrackerBoosting::create();
  else if (trackerType == trackerTypes[1])
    tracker = TrackerMIL::create();
  else if (trackerType == trackerTypes[2])
    tracker = TrackerKCF::create();
  else if (trackerType == trackerTypes[3])
    tracker = TrackerTLD::create();
  else if (trackerType == trackerTypes[4])
    tracker = TrackerMedianFlow::create();
  else if (trackerType == trackerTypes[5])
    tracker = TrackerGOTURN::create();
  else if (trackerType == trackerTypes[6])
    tracker = TrackerMOSSE::create();
  else if (trackerType == trackerTypes[7])
    tracker = TrackerCSRT::create();
  else {
    cout << "Incorrect tracker name" << endl;
    cout << "Available trackers are: " << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
      std::cout << " " << *it << endl;
  }
  return tracker;
}

enum class VideoSource {
    video_file = 0x0,
    web_camera = 0x1
};

struct mt_configurator {

    VideoSource src;
    string video_abs_path;
    size_t cam_id;
    size_t ver_res;
    size_t hor_res;
    string tracker_type;

    mt_configurator() {
        src = VideoSource::web_camera;
        video_abs_path = "/home/igor/temp/test_video.avi";
        cam_id = 0;
        ver_res = 240;
        hor_res = 320;
        tracker_type = "TLD";
    }
};

ostream& operator<<(ostream &os, mt_configurator &conf) {
    os << "Configurations: " << endl;
    os << "Video source: " << ((conf.src == VideoSource::video_file) ? "video file" : "web_Camera") << endl;
    if (conf.src == VideoSource::video_file)
        os << "Path: " << conf.video_abs_path << endl;
    else
        os << "Id: " << conf.cam_id << endl;
    os << "Vertical res: " << conf.ver_res << endl;
    os << "Horizontal res: " << conf.hor_res << endl;
    os << "Tracker type: " << conf.tracker_type << endl;
    return os;
}

void parse_args(mt_configurator &conf, stringstream &ss, int cnt) {
    cout << "SS content" << ss.str();

    string src;
    ss >> src;

    // src
    if (src == "video") {
        conf.src = VideoSource::video_file;
        if (cnt > 2)
            ss >> conf.video_abs_path;
    } else {
        conf.src = VideoSource::web_camera;
        if (cnt > 2)
            ss >> conf.cam_id;
    }

    if (cnt > 3) {
        ss >> conf.ver_res;
    }
    if (cnt > 4) {
        ss >> conf.hor_res;
    }
    if (cnt > 5) {
        ss >> conf.tracker_type;
    }
}

int main(int argc, char *argv[])
{
    mt_configurator config;
    if (argc > 1) {
        cout << "Parse args..." << endl;
        stringstream ss;
        for (int i = 1; i < argc; i++) {
            ss << string(argv[i]) << " ";
        }
        parse_args(config, ss, argc);
    } else {
        cout << "Default configuration..." << endl;
    }

    cout << config;

    cv::VideoCapture cap;
    if (config.src == VideoSource::video_file) {
        cap = VideoCapture(config.video_abs_path);
        if(!cap.isOpened()) {
          cout << "Error opening video file " << config.video_abs_path << endl;
          return 0;
        }
    } else {
        cap = VideoCapture(config.cam_id);
        if(!cap.isOpened()) {
          cout << "Error opening web-camera " << config.cam_id << endl;
          return 0;
        }
    }

      // Initialize MultiTracker with tracking algo
      vector<Rect> bboxes;

      Mat src_frame;
      Mat frame;
      Mat show_frame;


      // read first frame
      cap >> src_frame;
      resize(src_frame, frame, Size(config.hor_res, config.ver_res));


      // Get bounding boxes for first frame
      // selectROI's default behaviour is to draw box starting from the center
      // when fromCenter is set to false, you can draw box starting from top left corner
      bool showCrosshair = true;
      bool fromCenter = false;
      cout << "\n==========================================================\n";
      cout << "Press Escape to exit selection process" << endl;
      cout << "\n==========================================================\n";
      cv::selectROIs("MultiTracker", frame, bboxes, showCrosshair, fromCenter);

      // quit if there are no objects to track
      if(bboxes.size() < 1)
        return 0;

      vector<Scalar> colors;
      getRandomColors(colors, bboxes.size());

      // Specify the tracker type
      string trackerType = config.tracker_type;
      // Create multitracker
      Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();

      // Initialize multitracker
      for(int i=0; i < bboxes.size(); i++)
        multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));

      while(cap.isOpened())
      { LOG_DURATION("Processing");
        // get frame from the video
        cap >> src_frame;
        resize(src_frame, frame, Size(config.hor_res, config.ver_res));

        // Stop the program if reached end of video
        if (frame.empty()) break;

        //Update the tracking result with new frame
        multiTracker->update(frame);

        // Draw tracked objects
        for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
        {
          rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
        }


        resize(frame, show_frame, src_frame.size());
        // Show frame
        imshow("MultiTracker", show_frame);

        // quit on x button
        if  (waitKey(1) == 27) break;

       }
    return 0;
}
