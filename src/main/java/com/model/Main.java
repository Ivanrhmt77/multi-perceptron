package com.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        log.info("========================================");
        log.info("       MEMULAI TRAINING MODEL MLP       ");
        log.info("========================================");

        MultiPerceptronModel mlp = new MultiPerceptronModel();

        mlp.train();
        mlp.logFinalResults();

        log.info("\nTraining Selesai.");
        log.info("Log per epoch disimpan di: logs/epoch_log.csv");
        log.info("Hasil prediksi akhir disimpan di: logs/training_results.csv");
        log.info("========================================");
    }
}