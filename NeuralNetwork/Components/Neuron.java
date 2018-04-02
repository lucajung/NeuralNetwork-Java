package NeuralNetwork.Components;

import java.util.ArrayList;

/**
 * The Neuron is the smallest unit at the neural network.
 * It saves all connections to the next layes neurons,
 * its value and its error (backpropagation)
 *
 * @author Luca Jung
 * @version 3.0
 */
public class Neuron {

    private ArrayList<Double> connections = new ArrayList<>();
    private int activationFunction = 0;
    private Double value = null;
    private Double error = null;

    public void setValue(Double value){
        this.value = value;
    }

    public Double getValue(){
        return this.value;
    }

    public void addConnection(Double value){
        connections.add(value);
    }

    public void setConnection(int index, Double value){
        connections.set(index, value);
    }

    public void setActivationFunction(int activationFunction){
        this.activationFunction = activationFunction;
    }

    public int getActivationFunction(){
        return this.activationFunction;
    }

    public void setError(Double error){
        this.error = error;
    }

    public Double getError() {
        return this.error;
    }

    public Double getConnection(int index){
        if(index >= 0 && index < connections.size()){
            return connections.get(index);
        }
        else{
            return null;
        }
    }

    public int getNumberOfConnections(){
        return connections.size();
    }

}
