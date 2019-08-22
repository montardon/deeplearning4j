package com.techad;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 Load a Unet zoo model and try to train it on
 https://github.com/zhixuhao/unet.git
 membranes images (ISBI 2015 challenge)
 to get segmentation of input image
 */
public class TrainUnetWithMembraneImages {
    // Logger
    private static final Logger log = LoggerFactory.getLogger(TrainUnetWithMembraneImages.class);

    // Input image size 512x512x3 for this Unet model. It outputs 512x512x1 data.
    private static final int WIDTH = 512;
    private static final int HEIGHT = 512;
    private static final int CHANNELS = 3;

    public static void main(String[] args) throws IOException {

        String inputDataDirectory = "";
        if (args.length>0) {
            inputDataDirectory = args[0];
        } else {
            usage();
        }

        Random rng = new Random(1234);

        // Define cache location for downloaded models and data if necessary
        // By default cache is put in $HOME/.deeplearning4j repository
        DL4JResources.setBaseDirectory(new File(System.getProperty("user.dir")+"/cache"));

        // Load model from DL4J Zoo (model and weights)
        UNet unetModel = UNet.builder().build();

        int batchSize = 1; // not enough memory on device ? (4Gb)


        ComputationGraph model = null;
        try {
            // read saved model
            // model = ModelSerializer.restoreComputationGraph(new File("modelDL4J_unet"));
            // or instantiate pretrained zoo model
            model = (ComputationGraph) unetModel.initPretrained(PretrainedType.SEGMENT);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Print model vertices (debug)
        for (GraphVertex vertex: model.getVertices()) {
            System.out.println(vertex.getVertexName());
            if (vertex.getVertexName().equalsIgnoreCase("input_1")) {
                Layer layer = vertex.getLayer();
                if (layer != null) {
                    System.out.println(layer.toString());
                }
            }
        }

        // from Gitter deeplearning4j forum

        model = new TransferLearning.GraphBuilder(model)
            .removeVertexAndConnections("activation_23")
            .addLayer("output", new CnnLossLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "conv2d_23")
            .setOutputs("output")
            .build();

        // far too small learning rate
        model.setLearningRate(0.0000005f);

        long startTime = System.currentTimeMillis();

        // Define input images and labels
        // training and images and label /unet/data/membrane/train/img and
        // training and images and label /unet/data/membrane/train/label
        int numEpochs = 100;
        int numSamples = 30;
        try {
            File[] images = new File[numSamples];
            File[] labels = new File[numSamples];
            for (int i=0; i < numSamples; i++) {
                images[i] = new File(inputDataDirectory+"/membrane/train/image/"+i+".png");
                labels[i] = new File(inputDataDirectory+"/membrane/train/label/"+i+".png");
            }

            //Then add the StatsListener to collect this information from the network, as it trains
            model.setListeners(new PerformanceListener(30),new ScoreIterationListener(30));

            for (int epochs = 0; epochs < numEpochs; epochs++) {
                log.warn("Epoch "+epochs);
                for (int i=0; i <numSamples; i++) {
                    INDArray input = transformImageToBatch(images[i]);
                    INDArray answer = transformLabelToBatch(labels[i]);
                    model.fit(new INDArray[]{input}, new INDArray[]{answer});
                }
                System.gc();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Elapsed time="+(System.currentTimeMillis()-startTime)/1000.f+"s");
        try {
            ModelSerializer.writeModel(model,new File("modelDL4J_unet"),true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Train done.");
        System.out.println("Let's call exit()");
        System.exit(0);

    }

    public static void usage() {
        log.warn("Please call this program with a image filename as argument");
    }

    public static INDArray transformImageToBatch(File... imageFiles) throws IOException {
        INDArray imageBatch = Nd4j.create(DataType.FLOAT, imageFiles.length, CHANNELS  ,HEIGHT,WIDTH);
        int index = 0;
        NativeImageLoader loader = new NativeImageLoader(WIDTH,HEIGHT,CHANNELS);
        for (File imageFile: imageFiles) {
            INDArray arr = loader.asMatrix(imageFile);
//            log.warn(arr.shapeInfoToString());
            arr = arr.divi(255.0f).subi(0.5f);
            imageBatch.putRow(index++, arr);
        }
        return imageBatch;
    }

    public static INDArray transformLabelToBatch(File... imageFiles) throws IOException {
        INDArray imageBatch = Nd4j.create(DataType.FLOAT, imageFiles.length, 1  ,HEIGHT,WIDTH);
        int index = 0;
        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT,WIDTH,1);
        for (File imageFile: imageFiles) {
            INDArray arr = nativeImageLoader.asMatrix(imageFile);
//            log.warn(arr.shapeInfoToString());
            arr = arr.divi(255.0f);
            imageBatch.putRow(index++, arr);
        }
        return imageBatch;
    }

}
