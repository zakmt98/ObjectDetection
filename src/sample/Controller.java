package sample;


import java.awt.*;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;

public class Controller {
    public static String getRandomStr(int n)
    {

        String str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                + "abcdefghijklmnopqrstuvxyz";

        StringBuilder s = new StringBuilder(n);

        for (int i = 0; i < n; i++) {
            int index = (int)(str.length() * Math.random());
            s.append(str.charAt(index));
        }
        return s.toString();
    }
    private static List<String> getOutputNames(Net net) {



        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }


    @FXML
    public Button btn1, btn2;
    public Label label;


    public String imagePath;

    public String Img_output;

    private Desktop desktop = Desktop.getDesktop();

    public void ButtonSelectAction(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();

        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("JPG Image", "*.jpg"),
                new FileChooser.ExtensionFilter("PNG Image", "*.png"),
                new FileChooser.ExtensionFilter("JPEG Image", "*.jpeg")
        );
        File selectedImage = fileChooser.showOpenDialog(null);

        if(selectedImage != null) {
            imagePath = selectedImage.getAbsolutePath();
            label.setText(imagePath);
            System.out.println(label);
        }else {
            System.out.println("Image is not valid!");
        }
    }



    public void ButtonDetectAction(ActionEvent event) throws Exception {

        if(imagePath != null) {

                System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

                String modelWeights = "C:\\Users\\zakaria\\IdeaProjects\\object detection\\yolov3.weights";
                String modelConfiguration = "C:\\Users\\zakaria\\IdeaProjects\\object detection\\yolov3.cfg";
                String modelNames = "C:\\Users\\zakaria\\IdeaProjects\\object detection\\coco.names";


                ArrayList<String> classes = new ArrayList<>();
                FileReader file = new FileReader(modelNames);
                BufferedReader bufferedReader = new BufferedReader(file);
                String Line;
                while ((Line = bufferedReader.readLine()) != null) {
                    classes.add(Line);
                }
                bufferedReader.close();




                Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);


                Mat image1 = Imgcodecs.imread(label.getText());
                Size sz = new Size(416, 416);
                Mat blob = Dnn.blobFromImage(image1, 0.00392, sz, new Scalar(0), true, false);
                System.out.println(blob);
                net.setInput(blob);

                java.util.List<Mat> result = new ArrayList<>();
                java.util.List<String> outBlobNames = getOutputNames(net);

                net.forward(result, outBlobNames);


                outBlobNames.forEach(System.out::println);
                result.forEach(System.out::println);
                //System.out.println(result);

                float confThreshold = 0.5f;

                LinkedList<Integer> clsIds = new LinkedList<>();
                java.util.List<Float> confs = new ArrayList<>();
                List<Rect> rects = new ArrayList<>();




                for (int i = 0; i < result.size(); i++)
                {


                    Mat level = result.get(i);
                    for (int j = 0; j < level.rows(); ++j)
                    {
                        Mat row = level.row(j);
                        Mat scores = row.colRange(5, level.cols());

                        Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                        float confidence = (float)mm.maxVal;
                        org.opencv.core.Point classIdPoint = mm.maxLoc;

                        if (confidence > confThreshold)
                        {
                            int centerX = (int)(row.get(0,0)[0] * image1.cols());
                            int centerY = (int)(row.get(0,1)[0] * image1.rows());
                            int width   = (int)(row.get(0,2)[0] * image1.cols());
                            int height  = (int)(row.get(0,3)[0] * image1.rows());
                            int left    = centerX - width  / 2;
                            int top     = centerY - height / 2;


                            clsIds.addLast( (int)classIdPoint.x );
                            confs.add(confidence);
                            System.out.println(confidence);
                            rects.add(new Rect(left, top, width, height));

                        }
                    }
                }





                float nmsThresh = 0.5f;
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

                Rect[] boxesArray = rects.toArray(new Rect[0]);

                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();

                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);




                int [] ind = indices.toArray();

                for (int i = 0; i < ind.length; ++i)
                {
                    int idx = ind[i];
                    Rect box = boxesArray[idx];
                    Imgproc.rectangle(image1, box.tl(), box.br(), new Scalar(255,255,0), 5,20);

                    String label1 = classes.get(clsIds.get(i)).toString();


                    DecimalFormat df = new DecimalFormat("#.##");

                    label1=label1.concat(" "+ String.valueOf(df.format(confs.get(i)*100))+"%");

                    Imgproc.putText(image1,label1,new Point(box.x+30,box.y), 2, 2, new Scalar(0,0,255),2);

                    // System.out.println(box);
                }
                Img_output=getRandomStr(10);
                Imgcodecs.imwrite(Img_output+".png", image1);
            File file1 = new File("C:\\Users\\zakaria\\IdeaProjects\\FX_tut\\"+Img_output+".png");
            openFile(file1);







        }else {
            System.out.println("Invalid Image Path");
        }
    }


    private void openFile(File file) {
        try {
            desktop.open(file);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(
                    Level.SEVERE, null, ex
            );
        }
    }
}
