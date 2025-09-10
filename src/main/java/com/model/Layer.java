package com.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer {
    private List<Neuron> neurons;

    public Layer(int totalNeurons, int totalInputsPerNeuron, Random random) {
        neurons = new ArrayList<>();
        for (int i = 0; i < totalNeurons; i++) {
            neurons.add(new Neuron(totalInputsPerNeuron, random));
        }
    }

    /* --- Feed Forward --- */
    public List<Double> feedForward(double[] inputs) {
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            neuron.activate(inputs);
            outputs.add(neuron.getOutput());
        }
        return outputs;
    }

    /* --- Backward Error --- */
    public void calculateOutputDeltas(double[] target) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).calculateOutputDelta(target[i]);
        }
    }

    public void calculateHiddenDeltas(Layer nextLayer) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).calculateHiddenDelta(nextLayer, i);
        }
    }

    /* --- Weights Update --- */
    public void updateWeights(double[] lastInputs) {
        for (Neuron neuron : neurons) {
            neuron.updateWeights(lastInputs);
        }
    }

    /* --- Getters --- */
    public List<Neuron> getNeurons() {
        return neurons;
    }
}