package com.model.config;

public final class MLPConfig {
    private MLPConfig() {
    }

    public static final long RANDOM_SEED = 1337L;
    public static final int MAX_EPOCHS = 20000;
    public static final double LEARNING_RATE = 0.1;

    public static final double TARGET_ACCURACY = 1.0;
    public static final double PREDICTION_TOLERANCE = 0.05;

    public static final double[][] TRAINING_INPUTS = {
            { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
    };
    public static final double[][] TRAINING_TARGETS = {
            { 0 }, { 1 }, { 1 }, { 0 }
    };

    public static final int TOTAL_INPUTS = 2;
    public static final int TOTAL_HIDDEN_NEURONS = 2;
    public static final int TOTAL_OUTPUTS = 1;
}