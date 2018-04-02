package NeuralNetwork.Testing;

import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Training.DataSets;

import java.util.Arrays;

public class Testing {

    public static void main(String [] args) throws Exception
    {
        NeuralNetwork network = new NeuralNetwork(new int[]{2,4,1});
        DataSets trainingData = new DataSets();
        trainingData.addDataSet(new Double[]{0.2, 0.2}, new Double[]{0.4});
        trainingData.addDataSet(new Double[]{0.3, 0.4}, new Double[]{0.7});
        trainingData.addDataSet(new Double[]{0.7, 0.1}, new Double[]{0.8});
        network.connectNeuralNetworkRandomly();
        System.out.println(Arrays.toString(network.predict(new Double[]{0.2, 0.2})));
        System.out.println(Arrays.toString(network.predict(new Double[]{0.3, 0.4})));
        System.out.println(Arrays.toString(network.predict(new Double[]{0.3, 0.2})));
        System.out.println("Start training!");
        for (int i = 0; i < 70000; i++) {
            for (int j = 0; j < trainingData.size(); j++) {
                network.trainNeuralNetwork(trainingData.getDataSet(j)[0], trainingData.getDataSet(j)[1], 0.008);
            }
        }
        try {
            System.out.println(Arrays.toString(network.predict(new Double[]{0.2, 0.2})));
            System.out.println(Arrays.toString(network.predict(new Double[]{0.3, 0.4})));
            System.out.println(Arrays.toString(network.predict(new Double[]{0.3, 0.2})));
        } catch (Exception e){
            e.printStackTrace();
        }
    }

}
