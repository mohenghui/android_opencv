package zsc.androidstudy.android_opencv;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity
{

    private static final String CV_TAG = "openCV";

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initLoadOpenCV();

        Button button = (Button) findViewById(R.id.process_btn);
        button.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.im_ck2);
                Mat src = new Mat();
                Mat dst = new Mat();
                Utils.bitmapToMat(bitmap,src);
                Imgproc.cvtColor(src,dst, Imgproc.COLOR_BGRA2GRAY);
                Utils.matToBitmap(dst,bitmap);
                ImageView iv = (ImageView) findViewById(R.id.sample_img);
                iv.setImageBitmap(bitmap);
                src.release();
                dst.release();
            }
        });

    }

    //加载openCV本地库
    private void initLoadOpenCV()
    {
        boolean success = OpenCVLoader.initDebug();
        if (success)
        {
            System.out.println("loading success");
            Log.d("test", "initLoadOpenCVLibs:OpenCV加载成功!");
        }
        else
        {
            System.out.println("loading failed");
            Log.d("test", "initLoadOpenCVLibs:OpenCV加载失败!");
        }
    }
}