package com.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import com.model.config.MLPConfig;

public class Neuron {
    private List<Double> weights;
    private double bias;
    private double output;
    private double delta;

    public Neuron(int totalInputs, Random random) {
        weights = new ArrayList<>();
        bias = random.nextDouble() * 2.0 - 1.0;
        for (int i = 0; i < totalInputs; i++) {
            weights.add(random.nextDouble() * 2.0 - 1.0);
        }
    }

    /* --- Activation Function --- */
    private double accumulateNet(double[] inputs) {
        double net = bias;
        for (int i = 0; i < weights.size(); i++) {
            net += weights.get(i) * inputs[i];
        }
        return net;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void activate(double[] inputs) {
        output = sigmoid(accumulateNet(inputs));
    }

    /* --- Backward Error --- */
    private double sigmoidDerivative(double y) {
        return y * (1.0 - y);
    }

    public void calculateOutputDelta(double target) {
        double error = target - output;
        delta = error * sigmoidDerivative(output);
    }

    public void calculateHiddenDelta(Layer nextLayer, int neuronIndex) {
        double totalErrorFromNextLayer = 0.0;
        for (Neuron neuron : nextLayer.getNeurons()) {
            totalErrorFromNextLayer += neuron.getWeights().get(neuronIndex) * neuron.getDelta();
        }
        delta = totalErrorFromNextLayer * sigmoidDerivative(output);
    }

    /* --- Weights Update --- */
    public void updateWeights(double[] lastInputs) {
        for (int i = 0; i < weights.size(); i++) {
            double deltaWeight = MLPConfig.LEARNING_RATE * lastInputs[i] * delta;
            weights.set(i, weights.get(i) + deltaWeight);
        }
        bias += MLPConfig.LEARNING_RATE * delta;
    }

    /* --- Getters --- */
    public double getOutput() {
        return output;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public double getDelta() {
        return delta;
    }
}