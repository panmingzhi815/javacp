package org.pan;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_RGB2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

public class Main {
    private final static Logger LOGGER = LoggerFactory.getLogger(Main.class);
    private static opencv_objdetect.CascadeClassifier face_cascade;
    private static opencv_core.Mat temp = new opencv_core.Mat();
    private static OpenCVFrameConverter.ToMat convertToMat;

    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException {
        CanvasFrame canvasFrame = new CanvasFrame("Camera");
        canvasFrame.setSize(640, 480);
        canvasFrame.setVisible(true);

        FrameGrabber grabber = FrameGrabber.createDefault(0);
        grabber.start();

        face_cascade = new opencv_objdetect.CascadeClassifier("haarcascade_frontalface_alt2.xml");
        convertToMat = new OpenCVFrameConverter.ToMat();

        while (canvasFrame.isVisible()) {
            try {
                Frame frame = grabber.grab();

                if (frame.imageHeight == 0 || frame.imageWidth == 0) {
                    continue;
                }

                opencv_core.Mat convert = convertToMat.convert(frame);
                if (convert.empty()) {
                    continue;
                }
                detectFace(convert);
                canvasFrame.showImage(frame);
            } catch (Exception e) {
                LOGGER.error("采集相机异常", e);
            }
        }
        grabber.stop();
        System.exit(0);
    }

    public static opencv_core.RectVector detectFace(opencv_core.Mat image) {
        // 在图片中检测人脸
        opencv_core.RectVector faceDetections = new opencv_core.RectVector();
        cvtColor(image, temp, COLOR_RGB2GRAY);

        face_cascade.detectMultiScale(image, faceDetections,1.1, 2, 0,new opencv_core.Size(100,100),new opencv_core.Size());

        for (int i = 0; i < faceDetections.size(); i++) {
            opencv_core.Rect rect = faceDetections.get(i);
            opencv_imgproc.rectangle(image, new opencv_core.Point(rect.x() - 2, rect.y() - 2), new opencv_core.Point(rect.x()
                    + rect.width(), rect.y() + rect.height()), new opencv_core.Scalar(0, 0, 255, 0));
        }
        return faceDetections;
    }

}
