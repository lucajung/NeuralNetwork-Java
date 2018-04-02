package NeuralNetwork.Tools;

/**
 * NeuralNetworkMathFunctions are used by the neural network
 * in order to store activation functions and their derivatives.
 *
 * @author Luca Jung
 * @version 1.0
 */
public class NeuralNetworkMathFunctions {

    public final int SIGMOID_FUNCTION = 0;
    public final int RELU_FUNCTION = 1;
    public final int LEAKYRELU_FUNCTION = 2;

    public Double getValue(int function, Double x, boolean derivative){
        switch (function){
            case SIGMOID_FUNCTION:
                if(derivative){
                    return this.sigmoidDerivativeFunction(x);
                }
                else {
                    return this.sigmoidFunction(x);
                }
            case RELU_FUNCTION:
                if(derivative){
                    return this.reLUDerivativeFunction(x);
                }
                else {
                    return this.reLUFunction(x);
                }
            case LEAKYRELU_FUNCTION:
                if(derivative){
                    return this.leakyReLUDerivativeFunction(x);
                }
                else {
                    return this.leakyReLUFunction(x);
                }
        }
        return null;
    }

    private Double sigmoidFunction(Double x){
        return 1 / (1 + Math.exp(-x));
    }

    private Double sigmoidDerivativeFunction(Double x){
        return sigmoidFunction(x) * (1 - sigmoidFunction(x));
    }

    private Double reLUFunction(Double x){
        if(x > 0){
            return x;
        }
        else {
            return 0.0;
        }
    }

    private Double reLUDerivativeFunction(Double x){
        if(x > 0){
            return 1.0;
        }
        else {
            return 0.0;
        }
    }

    private Double leakyReLUFunction(Double x){
        if(x > 0){
            return x;
        }
        else {
            return 0.01 * x;
        }
    }

    private Double leakyReLUDerivativeFunction(Double x){
        if(x > 0){
            return 1.0;
        }
        else {
            return 0.01;
        }
    }

    public Double getRandomDouble(Double min, Double max){
        return min + Math.random() * (max - min);
    }

}