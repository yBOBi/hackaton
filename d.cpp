#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class DBSCAN {
public:
    DBSCAN(double eps, int min_samples) : eps_(eps), min_samples_(min_samples) {}

    std::vector<int> fit(const std::vector<cv::Point2f>& points) {
        std::vector<int> labels(points.size(), -1);
        int cluster_id = 0;

        for (int i = 0; i < points.size(); ++i) {
            if (labels[i] != -1) continue;

            std::vector<int> neighbors = region_query(points, i);
            if (neighbors.size() < min_samples_) {
                labels[i] = -1;  // Noise
            }
            else {
                expand_cluster(points, labels, i, neighbors, cluster_id);
                ++cluster_id;
            }
        }

        return labels;
    }

private:
    double eps_;
    int min_samples_;

    std::vector<int> region_query(const std::vector<cv::Point2f>& points, int point_idx) {
        std::vector<int> neighbors;
        for (int i = 0; i < points.size(); ++i) {
            if (cv::norm(points[point_idx] - points[i]) <= eps_) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }

    void expand_cluster(const std::vector<cv::Point2f>& points, std::vector<int>& labels,
        int point_idx, std::vector<int>& neighbors, int cluster_id) {
        labels[point_idx] = cluster_id;

        for (int i = 0; i < neighbors.size(); ++i) {
            int neighbor = neighbors[i];
            if (labels[neighbor] == -1) {
                labels[neighbor] = cluster_id;
                std::vector<int> new_neighbors = region_query(points, neighbor);
                if (new_neighbors.size() >= min_samples_) {
                    neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
                }
            }
        }
    }
};

cv::Mat translateImg(const cv::Mat& img, float offset_x, float offset_y) {
    cv::Mat trans_mat = (cv::Mat_<float>(2, 3) << 1, 0, offset_x, 0, 1, offset_y);
    cv::Mat result;
    cv::warpAffine(img, result, trans_mat, img.size());
    return result;
}

cv::Mat adjust_contrast_brightness(const cv::Mat& img, float contrast = 1.0, int brightness = 0) {
    brightness += static_cast<int>(std::round(255 * (1 - contrast) / 2));
    cv::Mat result;
    cv::addWeighted(img, contrast, img, 0, brightness, result);
    return result;
}

int main() {
    std::cout << "Введите номер видео 1-3 (ESC - закрыть видео): ";
    std::string video;
    std::cin >> video;

    cv::VideoCapture cap;
    double kontr;

    if (video == "2") {
        cap.open("./data/p2.mp4");
        kontr = 2.55;
    }
    else if (video == "3") {
        cap.open("./data/p3.mp4");
        kontr = 2.75;
    }
    else if (video == "4") {
        cap.open("./data/p4.mp4");
        kontr = 2.9;
    }
    else if (video == "5") {
        cap.open("./data/p5.mp4");
        kontr = 2.75;
    }
    else if (video == "22") {
        cap.open("./data/c2.mp4");
        kontr = 2.88;
    }
    else if (video == "33") {
        cap.open("./data/c3.mp4");
        kontr = 2.4;
    }
    else if (video == "11") {
        cap.open("./data/c1.mp4");
        kontr = 2.85;
    }
    else {
        cap.open("./data/c3.mp4");
        kontr = 2.4;
    }

    int fps = 60;
    cv::Ptr<cv::BackgroundSubtractor> subtractor = cv::createBackgroundSubtractorMOG2(100, 100, true);
    cv::Ptr<cv::DenseOpticalFlow> optical_flow = cv::optflow::createOptFlow_Farneback();
    int n = 100;
    int bb = 5;

    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03);
    std::vector<cv::Point2f> p0, p1;

    cv::Mat frame, prev_frame;

    while (true) {

        if (!cap.read(frame)) break;

        if (prev_frame.empty()) {
            cv::cvtColor(frame, prev_frame, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(prev_frame, prev_frame, cv::Size(bb, bb), 0);
            continue;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame, frame, cv::Size(bb, bb), 0);

        cv::Mat frame_arrow = frame.clone();
        cv::Mat output = frame.clone();

        cv::goodFeaturesToTrack(prev_frame, p0, n, 0.5, 7, cv::Mat(), 7);

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_frame, frame, p0, p1, status, err);

        std::vector<cv::Point2f> p0r;
        cv::calcOpticalFlowPyrLK(frame, prev_frame, p1, p0r, status, err, cv::Size(bb, bb), 2, criteria);

        std::vector<cv::Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++) {
            if (cv::norm(p0[i] - p0r[i]) < 1) {
                good_new.push_back(p1[i]);
            }
        }

        cv::Point2f sum(0, 0);
        for (const auto& p : good_new) {
            sum += p - p0[0];
        }
        cv::Point2f result = sum * (1.0 / good_new.size());

        cv::Mat prev_gray1 = translateImg(prev_frame, result.x, result.y);

        cv::Mat difference;
        cv::absdiff(prev_gray1, frame, difference);
        difference = adjust_contrast_brightness(difference(cv::Rect(10, 2, difference.cols - 12, difference.rows - 4)), kontr, 255);

        std::vector<std::vector<cv::Point>> contours_draw, contours_mask;
        cv::Mat image_edges;
        cv::threshold(difference, image_edges, 100, 255, cv::THRESH_BINARY);
        cv::findContours(image_edges, contours_draw, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        cv::findContours(image_edges, contours_mask, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int i = 0; i < contours_draw.size(); i++) {
            cv::drawContours(difference, contours_draw, i, cv::Scalar(0, 0, 0), 3);
        }

        for (int i = 2; i < contours_mask.size(); i++) {
            cv::fillConvexPoly(image_edges, contours_mask[i], cv::Scalar(0, 0, 0));
            cv::fillConvexPoly(difference, contours_mask[i], cv::Scalar(0, 255, 0));
        }

        if (!contours_mask.empty()) {
            std::vector<cv::Rect> bounding_boxes;
            for (const auto& contour : contours_mask) {
                bounding_boxes.push_back(cv::boundingRect(contour));
            }

            std::vector<cv::Point2f> boxes_points;
            for (const auto& box : bounding_boxes) {
                boxes_points.push_back(cv::Point2f(box.x, box.y));
                boxes_points.push_back(cv::Point2f(box.x + box.width, box.y + box.height));
            }

            float scale_x = 0.3f;
            for (auto& point : boxes_points) {
                point.x *= scale_x;
            }

            DBSCAN dbscan(20, 1);
            std::vector<int> labels = dbscan.fit(boxes_points);

            for (auto& point : boxes_points) {
                point.x /= scale_x;
            }

            std::vector<cv::Rect> merged_boxes;
            for (int cluster : labels) {
                std::vector<cv::Point2f> cluster_points;
                for (int i = 0; i < boxes_points.size(); ++i) {
                    if (labels[i] == cluster) {
                        cluster_points.push_back(boxes_points[i]);
                    }
                }

                if (!cluster_points.empty()) {
                    cv::Rect cluster_rect = cv::boundingRect(cluster_points);
                    merged_boxes.push_back(cluster_rect);
                }
            }

            for (const auto& box : merged_boxes) {
                cv::rectangle(frame_arrow, box, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("DBSCAN", frame_arrow);

        char k = cv::waitKey(30);
        if (k == 27) break;

        prev_frame = frame.clone();
    }

    return 0;
}
