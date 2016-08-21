package com.fatty.ml;

public class Main {

    public static void main(String[] args) {
        try {
            KangMisser misser = new KangMisser(0.0, 0.5);
            misser.miss("/Users/fatty/Downloads/ml_datasets_arff/abalone.arff", "./data/abalone.arff", 0.3);
        } catch (IllegalArgumentException | NullPointerException e) {
            System.out.println("Error occurs in main task. Details: " + e.getMessage());
        } catch (MissException e) {
            System.out.println("Miss data error. Details: " + e.getMessage());
        }
    }
}
