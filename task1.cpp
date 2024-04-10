#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    setlocale(LC_ALL, "Russian");

    int num_threads = 8;
    omp_set_num_threads(num_threads);

    auto start = chrono::steady_clock::now();

    CascadeClassifier face_cascade, eyes_cascade, smile_cascade;

    if (!face_cascade.load("C:/Users/James-Bond/Desktop/Распознование зрительных образов/Каскады/haarcascade_frontalface_default.xml") ||
        !eyes_cascade.load("C:/Users/James-Bond/Desktop/Распознование зрительных образов/Каскады/haarcascade_eye_tree_eyeglasses.xml") ||
        !smile_cascade.load("C:/Users/James-Bond/Desktop/Распознование зрительных образов/Каскады/haarcascade_smile.xml")) {
        printf("ошибка загрузки");
        return -1;
    }

    String videoPath = "C:/Users/James-Bond/Downloads/ZUA.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cout << "ошибка загрузки видео" << endl;
        return -1;
    }

    VideoWriter outputVideo("C:/Users/James-Bond/Downloads/ZUA_output_1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(CAP_PROP_FPS), Size(1280, 720));

    double startTime = omp_get_wtime();

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat gray_frame;
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        equalizeHist(gray_frame, gray_frame);

        GaussianBlur(frame, frame, Size(3, 3), 0);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray_frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

#pragma omp parallel sections shared(frame, faces)
        {
#pragma omp section
            {
                for (const Rect& face : faces) {
                    rectangle(frame, face, Scalar(255, 0, 0), 2);
                }
            }

#pragma omp section
            {
                for (const Rect& face : faces) {
                    Mat faceROI = gray_frame(face);
                    vector<Rect> eyes;
                    eyes_cascade.detectMultiScale(faceROI, eyes);
                    for (const Rect& eye : eyes) {
                        rectangle(frame, face.tl() + eye.tl(), face.tl() + eye.br(), Scalar(0, 255, 0), 2);
                    }
                }
            }

#pragma omp section
            {
                for (const Rect& face : faces) {
                    Mat faceROI = gray_frame(face);
                    vector<Rect> smile;
                    smile_cascade.detectMultiScale(faceROI, smile, 1.165, 35, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
                    for (const Rect& s : smile) {
                        rectangle(frame, face.tl() + s.tl(), face.tl() + s.br(), Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        namedWindow("распознавание лиц", WINDOW_NORMAL);
        resizeWindow("распознавание лиц", 600, 338);
        imshow("распознавание лиц", frame);
        outputVideo.write(frame);

        if (waitKey(1) == 27)
            break;

    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    cout << "количество потоков: " << num_threads << endl;
    cout << "время работы программы: " << elapsed_seconds.count() << "сек." << endl;

    cap.release();
    outputVideo.release();
    destroyAllWindows();

    return 0;
}
