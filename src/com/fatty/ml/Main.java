package com.fatty.ml;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class Main {

    public static void main(String[] args) {
        try {
            String originalFile = "/home/fatty/Mining/SA/abalone.arff";
            String missedFile = "./data/abalone2.arff";
            KangMisser misser = new KangMisser(0.0, 0.5);
            misser.miss(originalFile, missedFile, 0.3);
            Instances data = ConverterUtils.DataSource.read(missedFile);
            System.out.println(data.numInstances());
            System.out.println(data.classIndex());
        } catch (IllegalArgumentException | NullPointerException e) {
            System.out.println("Error occurs in main task. Details: " + e.getMessage());
        } catch (MissException e) {
            System.out.println("Miss data error. Details: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("Unhandled error. Details: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
