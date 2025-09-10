package com.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.model.config.MLPConfig;

public class MultiPerceptronModel {
    private List<Layer> layers;
    private int totalEpoch;

    private static final Logger log = LoggerFactory.getLogger(MultiPerceptronModel.class);
    private static final Logger epochLog = LoggerFactory.getLogger("EPOCHDATA");
    private static final Logger resultsLog = LoggerFactory.getLogger("CSVDATA");

    public MultiPerceptronModel() {
        layers = new ArrayList<>();
        Random random = new Random(MLPConfig.RANDOM_SEED);
        layers.add(new Layer(MLPConfig.TOTAL_HIDDEN_NEURONS, MLPConfig.TOTAL_INPUTS, random));
        layers.add(new Layer(MLPConfig.TOTAL_OUTPUTS, MLPConfig.TOTAL_HIDDEN_NEURONS, random));
        log.info("Model created with {} hidden neurons.", MLPConfig.TOTAL_HIDDEN_NEURONS);
    }

    /* --- Train Model --- */
    public void train() {
        epochLog.info("Epoch,MSE,Accuracy");

        for (int epoch = 0; epoch < MLPConfig.MAX_EPOCHS; epoch++) {
            double sse = 0.0;
            int correctPredictions = 0;

            for (int i = 0; i < MLPConfig.TRAINING_INPUTS.length; i++) {
                double[] inputs = MLPConfig.TRAINING_INPUTS[i];
                double[] targets = MLPConfig.TRAINING_TARGETS[i];

                List<Double> outputs = feedForward(inputs);
                backpropagate(targets);
                updateWeights(inputs);

                double prediction = outputs.get(0);
                double target = targets[0];
                double error = target - prediction;
                sse += error * error;

                if (Math.abs(error) <= MLPConfig.PREDICTION_TOLERANCE) {
                    correctPredictions++;
                }
            }

            this.totalEpoch = epoch + 1;

            double mse = sse / MLPConfig.TRAINING_INPUTS.length;
            double accuracy = (double) correctPredictions / MLPConfig.TRAINING_INPUTS.length;

            epochLog.info("{},{},{}", this.totalEpoch, mse, accuracy);

            if (this.totalEpoch % 1000 == 0) {
                log.info("Epoch {}/{} -> MSE: {}, Accuracy: {}%", this.totalEpoch, MLPConfig.MAX_EPOCHS,
                        String.format("%.8f", mse), String.format("%.2f", accuracy * 100));
            }

            if (accuracy >= MLPConfig.TARGET_ACCURACY) {
                log.info("Epoch {}/{} -> MSE: {}, Accuracy: {}%", this.totalEpoch, MLPConfig.MAX_EPOCHS,
                        String.format("%.8f", mse), String.format("%.2f", accuracy * 100));
                log.info("--- Early stopping! Target accuracy tercapai di epoch {}. ---", this.totalEpoch);
                break;
            }
        }
    }

    /* --- Show Final Results --- */
    public void logFinalResults() {
        log.info("================ HASIL PREDIKSI AKHIR ================");
        resultsLog.info("Input1,Input2,Target,Prediction,Error");
        for (int i = 0; i < MLPConfig.TRAINING_INPUTS.length; i++) {
            double[] input = MLPConfig.TRAINING_INPUTS[i];
            double[] target = MLPConfig.TRAINING_TARGETS[i];

            List<Double> predictionList = feedForward(input);
            double prediction = predictionList.get(0);

            log.info("Input: [{}, {}] -> Target: {}, Prediksi: {}", input[0], input[1], target[0],
                    String.format("%.5f", prediction));

            double error = target[0] - prediction;
            String csvLine = String.format("%.1f,%.1f,%.1f,%.5f,%.5f", input[0], input[1], target[0], prediction,
                    error);
            resultsLog.info(csvLine);
        }
        log.info("======================================================");
    }

    /* --- Feed Forward --- */
    public List<Double> feedForward(double[] inputs) {
        double[] currentInputs = inputs;
        for (Layer layer : layers) {
            List<Double> outputs = layer.feedForward(currentInputs);
            currentInputs = outputs.stream().mapToDouble(d -> d).toArray();
        }
        List<Double> finalOutputs = new ArrayList<>();
        for (double output : currentInputs) {
            finalOutputs.add(output);
        }
        return finalOutputs;
    }

    /* --- Backpropagation --- */
    public void backpropagate(double[] targets) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            if (i == layers.size() - 1) {
                layer.calculateOutputDeltas(targets);
            } else {
                Layer nextLayer = layers.get(i + 1);
                layer.calculateHiddenDeltas(nextLayer);
            }
        }
    }

    /* --- Weights Update --- */
    public void updateWeights(double[] initialInputs) {
        double[] inputs = initialInputs;
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            if (i > 0) {
                Layer previousLayer = layers.get(i - 1);
                inputs = new double[previousLayer.getNeurons().size()];
                for (int j = 0; j < inputs.length; j++) {
                    inputs[j] = previousLayer.getNeurons().get(j).getOutput();
                }
            }
            layer.updateWeights(inputs);
        }
    }

    /* --- Getters --- */
    public int getTotalEpoch() {
        return totalEpoch;
    }
}