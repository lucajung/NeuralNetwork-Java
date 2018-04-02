package NeuralNetwork;

import NeuralNetwork.Components.Layer;
import NeuralNetwork.Exceptions.NeuralNetworkException;
import NeuralNetwork.Tools.NeuralNetworkMathFunctions;

import java.util.ArrayList;


/**
 * Basic Neural Network with
 * integrated backpropagation
 *
 * @author Luca Jung
 * @version 3.2
 */
public class NeuralNetwork{

    private ArrayList<Layer> neuralNetwork = new ArrayList<>();
    private ArrayList<Double> neuralNetworkBias = new ArrayList<>();
    private NeuralNetworkMathFunctions mathFunctions = new NeuralNetworkMathFunctions();
    private boolean isConnected = false;

    public NeuralNetwork(int[] networkSize){
        for (int i = 0; i < networkSize.length; i++) {
            //add network layer
            neuralNetwork.add(new Layer(networkSize[i]));

            //add connection
            if(i > 0){
                neuralNetworkBias.add(null);
                for (int neuron = 0; neuron < neuralNetwork.get(i - 1).getNumberOfNeurons(); neuron++) {
                    for (int nextLayerNumberOfNeurons = 0; nextLayerNumberOfNeurons < neuralNetwork.get(i).getNumberOfNeurons(); nextLayerNumberOfNeurons++) {
                        neuralNetwork.get(i - 1).getNeuron(neuron).addConnection(null);
                    }
                }
            }
        }
        for (int neuron = 0; neuron < neuralNetwork.get(neuralNetwork.size() - 1).getNumberOfNeurons(); neuron++) {
            this.setActivationFunction(neuralNetwork.size() - 1, neuron, mathFunctions.LEAKYRELU_FUNCTION);
        }
    }

    public void connectNeuralNetworkRandomly(Double min, Double max) {
        for (int i = 1; i < neuralNetwork.size(); i++) {
            neuralNetworkBias.set(i - 1, mathFunctions.getRandomDouble(min,max));
            for (int neuron = 0; neuron < neuralNetwork.get(i - 1).getNumberOfNeurons(); neuron++) {
                for (int nextLayerNumberOfNeurons = 0; nextLayerNumberOfNeurons < neuralNetwork.get(i).getNumberOfNeurons(); nextLayerNumberOfNeurons++) {
                    Double randomDouble = mathFunctions.getRandomDouble(min,max);
                    neuralNetwork.get(i - 1).getNeuron(neuron).setConnection(nextLayerNumberOfNeurons, randomDouble);
                }
            }
        }
        this.setConnected(true);
    }

    public void connectNeuralNetworkRandomly() {
        this.connectNeuralNetworkRandomly(-0.8,0.8);
    }

    public Double[] predict(Double[] input) throws Exception {
        if(input.length == neuralNetwork.get(0).getNumberOfNeurons()){
            if(this.isNetworkReady()){
                //set input
                for (int inputNeuron = 0; inputNeuron < neuralNetwork.get(0).getNumberOfNeurons(); inputNeuron++) {
                    neuralNetwork.get(0).getNeuron(inputNeuron).setValue(input[inputNeuron]);
                }

                //calculate
                for (int layer = 1; layer < neuralNetwork.size(); layer++) {
                    for (int currentNeuron = 0; currentNeuron < neuralNetwork.get(layer).getNumberOfNeurons(); currentNeuron++) {
                        Double newValue = neuralNetworkBias.get(layer - 1);
                        for (int previousNeuron = 0; previousNeuron < neuralNetwork.get(layer - 1).getNumberOfNeurons(); previousNeuron++) {
                            newValue += neuralNetwork.get(layer - 1).getNeuron(previousNeuron).getConnection(currentNeuron) * neuralNetwork.get(layer - 1).getNeuron(previousNeuron).getValue();
                        }
                        newValue = mathFunctions.getValue(neuralNetwork.get(layer).getNeuron(currentNeuron).getActivationFunction(), newValue, false);
                        neuralNetwork.get(layer).getNeuron(currentNeuron).setValue(newValue);
                    }
                }

                Double[] output = new Double[neuralNetwork.get(neuralNetwork.size() - 1).getNumberOfNeurons()];
                for (int i = 0; i < output.length; i++) {
                    output[i] = neuralNetwork.get(neuralNetwork.size() - 1).getNeuron(i).getValue();
                }

                return output;

            }
            else{
                throw new NeuralNetworkException("Neural Network is not ready to use");
            }
        }
        else{
            throw new NeuralNetworkException("Unexpected input size");
        }
    }

    public void trainNeuralNetwork(Double[] input, Double[] target, Double learningRate) throws Exception {
        predict(input);
        calculateError(target);
        addDeltaToWeights(learningRate);
    }

    private void calculateError(Double[] target){
        for (int layer = neuralNetwork.size() - 1; layer >= 0; layer--) {
            if (layer == neuralNetwork.size() - 1){
                for (int neuron = 0; neuron < neuralNetwork.get(layer).getNumberOfNeurons(); neuron++) {
                    neuralNetwork.get(layer).getNeuron(neuron).setError(
                            mathFunctions.getValue(
                                    neuralNetwork.get(layer).getNeuron(neuron).getActivationFunction(),
                                    neuralNetwork.get(layer).getNeuron(neuron).getValue(),
                                    true
                            ) * (neuralNetwork.get(layer).getNeuron(neuron).getValue() - target[neuron])
                    );
                }
            }
            else {
                for (int neuron = 0; neuron < neuralNetwork.get(layer).getNumberOfNeurons(); neuron++) {
                    Double sum = 0.0;
                    for (int nextLayerNeuron = 0; nextLayerNeuron < neuralNetwork.get(layer + 1).getNumberOfNeurons(); nextLayerNeuron++) {
                        sum += neuralNetwork.get(layer + 1).getNeuron(nextLayerNeuron).getError() * neuralNetwork.get(layer).getNeuron(neuron).getConnection(nextLayerNeuron);
                    }
                    neuralNetwork.get(layer).getNeuron(neuron).setError(
                            mathFunctions.getValue(
                                    neuralNetwork.get(layer).getNeuron(neuron).getActivationFunction(),
                                    neuralNetwork.get(layer).getNeuron(neuron).getValue(),
                                    true
                            ) * sum
                    );
                }
            }
        }
    }

    private void addDeltaToWeights(double learningRate){
        for (int layer = 1; layer < neuralNetwork.size(); layer++) {
            for (int neuron = 0; neuron < neuralNetwork.get(layer).getNumberOfNeurons(); neuron++) {
                for (int previousLayerNeuron = 0; previousLayerNeuron < neuralNetwork.get(layer - 1).getNumberOfNeurons(); previousLayerNeuron++) {
                    Double delta = -learningRate * neuralNetwork.get(layer - 1).getNeuron(previousLayerNeuron).getValue() * neuralNetwork.get(layer).getNeuron(neuron).getError();
                    neuralNetwork.get(layer - 1).getNeuron(previousLayerNeuron).setConnection(neuron, neuralNetwork.get(layer - 1).getNeuron(previousLayerNeuron).getConnection(neuron) + delta);
                }
            }
        }
    }

    private void setConnected(boolean state){
        this.isConnected = state;
    }

    private boolean isNetworkReady(){
        if(isConnected && neuralNetwork.size() > 1){
            return true;
        }
        else{
            return false;
        }
    }

    public int getNetworkLayerSize(){
        return neuralNetwork.size();
    }

    public int getNumberOfNeurons(int layer){
        if(layer >= 0 && layer < getNetworkLayerSize()){
            return neuralNetwork.get(layer).getNumberOfNeurons();
        }
        else {
            return 0;
        }
    }

    public int getNumberOfConnections(int layer, int neuron){
        if(layer >= 0 && layer < getNetworkLayerSize()){
            if(neuron >= 0 && neuron < getNumberOfNeurons(layer)){
                return neuralNetwork.get(layer).getNeuron(neuron).getNumberOfConnections();
            }
        }
        return 0;
    }

    public void setActivationFunction(int layer, int neuron, int function){
        neuralNetwork.get(layer).getNeuron(neuron).setActivationFunction(function);
    }

    public void setConnection(int layer, int neuron, int connection, double connectionValue){
        neuralNetwork.get(layer).getNeuron(neuron).setConnection(connection, connectionValue);
    }

    public Double getConnection(int layer, int neuron, int connection){
        return neuralNetwork.get(layer).getNeuron(neuron).getConnection(connection);
    }

}