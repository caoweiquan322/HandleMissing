package com.fatty.ml;

import com.fatty.Helper;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.HashMap;
import java.util.Random;

public class Main {
    public static HashMap<String, Integer> createDataSetClassIndex() {
        HashMap<String, Integer> dataSetClassIndex = new HashMap<>();
        // Classification problems.
        dataSetClassIndex.put("iris", -1);
        dataSetClassIndex.put("wine", 0);
        dataSetClassIndex.put("sonar", -1);
        dataSetClassIndex.put("glass", -1);
        dataSetClassIndex.put("ionosphere", -1);
        dataSetClassIndex.put("pima", -1);
        dataSetClassIndex.put("satellite", -1);
        dataSetClassIndex.put("shuttle", -1);

        // Regression problems.
        // Todo: Add regression problems mapping.
        return dataSetClassIndex;
    }

    public static void main(String[] args) {
        try {
            HashMap<String, Integer> dataSetClassIndex = createDataSetClassIndex();

            String originalFile = "/home/fatty/Mining/SA/wine.arff";
            String missedFile = "./data/wine.arff";
            Misser misser = new KangMisser(0.0, 0.5);
            misser.miss(originalFile, missedFile, 0.6, dataSetClassIndex.get("wine"));
            Instances data = ConverterUtils.DataSource.read(missedFile);
            Helper.setDataSetClassIndex(data, dataSetClassIndex.get("wine"));

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(new J48(), data, 10, new Random());
            System.out.println(eval.toSummaryString());

            Instances imputedData = new MEIImputer().impute(data, dataSetClassIndex.get("wine"));
            ConverterUtils.DataSink.write("./data/wine_impute.arff", imputedData);
            Evaluation eval2 = new Evaluation(imputedData);
            eval2.crossValidateModel(new J48(), imputedData, 10, new Random());
            System.out.println(eval.toSummaryString());
        } catch (IllegalArgumentException | NullPointerException e) {
            System.out.println("Error occurs in main task. Details: " + e.getMessage());
        } catch (MissException e) {
            System.out.println("Miss data error. Details: " + e.getMessage());
        } catch (ImputeException e) {
            System.out.println("Impute data error. Details: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("Unhandled error. Details: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
