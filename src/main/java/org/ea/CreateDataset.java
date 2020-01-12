package org.ea;

import com.google.protobuf.ByteString;
import org.tensorflow.example.*;
import org.tensorflow.hadoop.util.TFRecordWriter;

import javax.imageio.IIOException;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class CreateDataset {

    public static void runFile(int label, TFRecordWriter tfWriter, File file) throws Exception {
        BufferedImage bi = null;
        try {
            bi = ImageIO.read(file);
        } catch(IIOException ie) {
            return;
        }
        if (bi == null) return;

        int width = bi.getWidth();
        int height = bi.getHeight();
        int depth = 3;

        Features features = Features.newBuilder()
                .putFeature("width", getIntFeature(width))
                .putFeature("height", getIntFeature(height))
                .putFeature("depth", getIntFeature(depth))
                .putFeature("label", getIntFeature(label))
                .putFeature("image_raw", getImageFeature(bi))
                .build();

        Example example = Example.newBuilder().setFeatures(features).build();

        tfWriter.write(example.toByteArray());
    }

    protected static Feature getImageFeature(BufferedImage orgImg) throws Exception {
        BufferedImage bi = new BufferedImage(orgImg.getWidth(), orgImg.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D)bi.getGraphics();
        g.drawImage(orgImg, 0, 0, null);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(bi, "jpeg", baos);
        baos.flush();
        ByteString byteString = ByteString.copyFrom(baos.toByteArray());
        baos.close();
        BytesList bytesList = BytesList.newBuilder().addValue(byteString).build();
        Feature text = Feature.newBuilder().setBytesList(bytesList).build();

        return text;
    }

    protected static Feature getIntFeature(int val) {
        Int64List int64List = Int64List.newBuilder().addValue(val).build();
        Feature intFeature = Feature.newBuilder().setInt64List(int64List).build();
        return intFeature;
    }

    public static void main(String[] args) {

        try {
            FileOutputStream fosTrain = new FileOutputStream("petimages2.tfrecords");
            DataOutputStream dosTrain = new DataOutputStream(fosTrain);
            TFRecordWriter tfWriterTrain = new TFRecordWriter(dosTrain);

            File dir = new File("PetImages");
            int label = 1;
            for(File imageDir : dir.listFiles()) {
                for(File f : imageDir.listFiles()) {
                    if(!f.isFile()) continue;
                    runFile(label, tfWriterTrain, f);
                }
                label++;
            }
            dosTrain.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
