package com.fatty.ml;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.FilteredNeighbourSearch;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;

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

    public static double evaluateDataSet(Instances instances, Classifier classifier) throws Exception {
        int numFold = 10;
        double performance = 0;
        final int NUM_TEST = 1;
        for (int i=0; i<NUM_TEST; ++i) {
            Evaluation eval = new Evaluation(instances);
            eval.crossValidateModel(classifier, instances, numFold, new Random());
            performance += eval.errorRate();
        }
        return 1.0-performance/NUM_TEST;

    }

    public static void main(String[] args) {
        try {
            HashMap<String, Integer> dataSetClassIndex = createDataSetClassIndex();
            double[] missingRatios = {0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};
            int missRepeat = 30;

            String dataSetName = "iris";
            String originalFile = "/home/fatty/Mining/SA/" + dataSetName + ".arff";
            int classIndex = dataSetClassIndex.get(dataSetName);
            Instances original = ConverterUtils.DataSource.read(originalFile);
            original.setClassIndex(original.numAttributes()-1);

            if (false) {
                Classifier classifier = new LLR(20);
                classifier.buildClassifier(original);
                int correct = 0;
                for (int i=0; i<original.numInstances(); ++i) {
                    Instance instance = original.get(i);
                    double oriX = instance.classValue();
                    instance.setClassMissing();
                    double x = classifier.classifyInstance(instance);
                    System.out.println("==> " + ((int)x) + ": " + ((int)oriX));
                    if ((int)x == ((int)oriX))
                        correct++;
                }
                System.out.println("Correctness: " + correct/1.0/original.numInstances());
                return;
            }

            Misser misser = new KangMisser(0.0, 0.5);

            for (double missingRatio : missingRatios) {
                double correctRate = 0.0;
                for (int i=0; i<missRepeat; ++i) {
                    Instances missed = misser.miss(original, missingRatio, classIndex);
                    Instances meiImputed = new MEIImputer().impute(missed, classIndex);
                    IBk classifier = new FastLLR();
                    classifier.setNearestNeighbourSearchAlgorithm(new FilteredNeighbourSearch());
                    classifier.setKNN(20);
                    correctRate += evaluateDataSet(missed, classifier);
                }
                correctRate /= missRepeat;
                System.out.println("Accuracy for missing ratio " + missingRatio + ": " + correctRate);
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
