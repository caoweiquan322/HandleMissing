package com.fatty.ml;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.FilteredNeighbourSearch;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddUserFields;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Normalize;

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
        dataSetClassIndex.put("wpbc", -1);
        dataSetClassIndex.put("stock", -1);
        dataSetClassIndex.put("abalone", -1);
        dataSetClassIndex.put("cpu_act", -1);
        dataSetClassIndex.put("bank8FM", -1);
        dataSetClassIndex.put("bank32NH", -1);
        dataSetClassIndex.put("kin8nm", -1);
        dataSetClassIndex.put("puma8NH", -1);
        dataSetClassIndex.put("puma32H", -1);
        dataSetClassIndex.put("cal_housing", -1);
        // dataSetClassIndex.put("house8L", -1);

        return dataSetClassIndex;
    }

    public static double evaluateDataSet(Instances instances, Classifier classifier) throws Exception {
        int numFold = 10;
        double performance = 0;
        final int NUM_TEST = 1;
        for (int i=0; i<NUM_TEST; ++i) {
            Evaluation eval = new Evaluation(instances);
            eval.crossValidateModel(classifier, instances, numFold, new Random());
            if (instances.classAttribute().isNominal())
                performance += eval.errorRate();
            else
                performance += eval.rootMeanSquaredError();
        }
        return performance/NUM_TEST;

    }

    public static void main(String[] args) {
        try {
            HashMap<String, Integer> dataSetClassIndex = createDataSetClassIndex();
            double[] missingRatios = {0.0, 0.2, 0.5};//{0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};
            int missRepeat = 5;

            String dataSetName = "cpu_act";
            String originalFile = "/home/fatty/Mining/SA/" + dataSetName + ".arff";
            int classIndex = dataSetClassIndex.get(dataSetName);
            Instances original = ConverterUtils.DataSource.read(originalFile);
            original.setClassIndex(dataSetClassIndex.get(dataSetName));

            // Add two noise column.
            System.out.println("Original has " + original.numAttributes() + " attributes.");
            Filter add1 = new Add();
            add1.setInputFormat(original);
            String options = "-T NUM -N noiseA -C first";
            add1.setOptions(Utils.splitOptions(options));
            original = Filter.useFilter(original, add1);

            Filter add2 = new Add();
            add2.setInputFormat(original);
            options = "-T NUM -N noiseB -C first";
            add2.setOptions(Utils.splitOptions(options));
            original = Filter.useFilter(original, add2);
            System.out.println("Original has " + original.numAttributes() + " attributes.");

            Random r = new Random();
            for (Instance line : original) {
                line.setValue(0, r.nextDouble()*10);
                line.setValue(1, r.nextDouble()*100);
            }

            Filter normalize = new Normalize();
            normalize.setInputFormat(original);
            original = Filter.useFilter(original, normalize);


            Misser misser = new KangMisser(0.0, 0.5);

            for (double missingRatio : missingRatios) {
                double performance = 0.0;
                for (int i=0; i<missRepeat; ++i) {
                    Instances missed = misser.miss(original, missingRatio, classIndex);
//                    Instances meiImputed = new MEIImputer().impute(missed, classIndex);
//                    IBk classifier = new FastLLR();
//                    classifier.setNearestNeighbourSearchAlgorithm(new FilteredNeighbourSearch());
//                    classifier.setKNN(20);
                    Classifier a = new LLR(8);
                    Classifier b = new WLLR(8);
                    Classifier c = new LinearRegression();
                    performance += evaluateDataSet(missed, b);
                }
                performance /= missRepeat;
                System.out.println("Accuracy for missing ratio " + missingRatio + ": " + performance);
            }

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
