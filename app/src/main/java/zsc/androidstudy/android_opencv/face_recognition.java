package zsc.androidstudy.android_opencv;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.android.JavaCamera2View;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class face_recognition extends AppCompatActivity implements
        CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {
    private CameraBridgeViewBase cameraView;
    private CascadeClassifier classifier;
    private Mat mGray;
    private Mat mRgba;
    private int mAbsoluteFaceSize = 0;
    private boolean isFrontCamera;
    // 手动装载openCV库文件，以保证手机无需安装OpenCV Manager   不加这里将导致无法初始化 级联分类器（开机闪退）
    static {
        System.loadLibrary("opencv_java3");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, 1);
        }
        initWindowSettings();
        setContentView(R.layout.face_recognition);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this); // 设置相机监听
        initClassifier();
        cameraView.enableView();
        Button switchCamera = (Button) findViewById(R.id.switch_camera);
        switchCamera.setOnClickListener(this); // 切换相机镜头，默认后置
    }
    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.switch_camera:
                cameraView.disableView();
                if (isFrontCamera) {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                    isFrontCamera = false;
                } else {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                    isFrontCamera = true;
                }
                cameraView.enableView();
                break;
            default:
        }
    }
    // 初始化窗口设置, 包括全屏、横屏、常亮
    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED);
    }
    // 初始化人脸级联分类器，必须先初始化
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.lbpcascade_frontalface_improved);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface_improved.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }
    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }
    @Override
    // 这里执行人脸检测的逻辑, 根据OpenCV提供的例子实现(face-detection)
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        // 翻转矩阵以适配前后置摄像头
        if (isFrontCamera) {
            Core.rotate(mRgba, mRgba, Core.ROTATE_90_COUNTERCLOCKWISE);
            Core.rotate(mGray, mGray, Core.ROTATE_90_COUNTERCLOCKWISE);
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);
        } else {
            //如果发现后摄出现了镜像  把下面的注释打开即可  魅族不需要
            Core.rotate(mRgba, mRgba, Core.ROTATE_90_COUNTERCLOCKWISE);
            Core.rotate(mGray, mGray, Core.ROTATE_90_COUNTERCLOCKWISE);
            Core.flip(mRgba, mRgba, -1);
            Core.flip(mGray, mGray, -1);
        }
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        MatOfRect faces = new MatOfRect();
        /**
         * @Param image 输入数据
         * @Param object 检测到的对象的数据的总容器，对象的数据以Rect方式保存
         * @Param scaleFactor 尺度变换的比率 ，基本在 l.05 ~ 1.2 之间
         * @Param minNeighbors 领域内符合条件的个数，满足个数才会采纳。
         * @param flags 默认为0（opencv2中的参数）
         * @Param minSize 检测的最小范围
         * @Param maxSize 检测的最大范围
         */
        if (classifier != null)
            classifier.detectMultiScale(mGray, faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        for (Rect faceRect : facesArray)
            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);
        return mRgba;
//        Mat dst = inputFrame.rgba();
//        //Log.e(TAG,"orientation : " + this.getRequestedOrientation());
//
//        // 90度 翻转
//        if(!isFrontCamera)
//            Core.rotate(dst, dst, Core.ROTATE_90_CLOCKWISE);
//        else
//        {
//            Core.rotate(dst, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
//            Core.flip(dst, dst, 1);
//        }
//
//        // Full Screen Preview
//        Imgproc.resize(dst,dst,new Size(cameraView.getWidth(), cameraView.getHeight()),0.0D,0.0D,0);
////        process_image(dst);
//        return dst;
    }
    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0 &&
                        grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                } else {
                    Toast.makeText(this, "权限拒绝", Toast.LENGTH_SHORT).show();
                }
        }
    }
}