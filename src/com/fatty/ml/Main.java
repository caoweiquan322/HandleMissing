package com.fatty.ml;

import com.fatty.Helper;
import com.fatty.ml.unimputed.UnimputedDecisionTable;
import com.fatty.ml.unimputed.UnimputedJ48;
import com.fatty.ml.unimputed.UnimputedSMO;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SystemInfo;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class Main {
    public static HashMap<String, Integer> createDataSetClassIndex() {
        HashMap<String, Integer> dataSetClassIndex = new HashMap<>();
        // Classification problems.
        dataSetClassIndex.put("iris", -1);
        dataSetClassIndex.put("wine", 0); // Wine should be handled in another way to add noise columns.
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
            //System.out.println("Unclassified: " + eval.unclassified());
            if (instances.classAttribute().isNominal())
                performance += eval.errorRate();
            else
                performance += eval.rootMeanSquaredError();
        }
        return performance/NUM_TEST;

    }

    public static Instances addNoiseColumns(Instances original, int numToAdd, int classIndex) throws Exception {
        // Check input parameters.
        if (numToAdd <= 0) {
            return original;
        }

        System.out.println("Original has " + original.numAttributes() + " attributes.");
        for (int i=0; i<numToAdd; ++i) {
            Filter add = new Add();
            add.setInputFormat(original);
            String options = classIndex<0 ? ("-T NUM -N fatty_noise"+i+" -C first") : ("-T NUM -N fatty_noise"+i+" -C last");
            add.setOptions(Utils.splitOptions(options));
            original = Filter.useFilter(original, add);
        }

        System.out.println("Original has " + original.numAttributes() + " attributes.");

        List<Integer> noises = new ArrayList<>();
        for (int i=0; i<original.numAttributes(); ++i) {
            if (original.attribute(i).name().startsWith("fatty_noise"))
                noises.add(i);
        }

        Random r = new Random();
        for (Instance line : original) {
            for (int i: noises)
                line.setValue(i, r.nextDouble());
        }
        return original;
    }

    public static Instances normalize(Instances original) throws Exception {
        Filter normalize = new Normalize();
        normalize.setInputFormat(original);
        return Filter.useFilter(original, normalize);
    }

    public static Instances nominalToNumeric(Instances original) throws Exception {
        Filter nominalFilter = new NominalToBinary();
        nominalFilter.setInputFormat(original);
        return Filter.useFilter(original, nominalFilter);
    }

    public static double checkMissingRatio(Instances dataset) {
        int numMissed = 0;
        for (Instance inst : dataset) {
            if (inst.hasMissingValue()) ++numMissed;
        }
        return numMissed/1.0/dataset.numInstances();
    }

    public static void main(String[] args) throws Exception {
        try {
            // The global map to use.
            HashMap<String, Integer> dataSetClassIndex = createDataSetClassIndex();

            // Specify the dataset.
            String dataSetName = "iris";
            int numInterruptColumns = 0;
            String originalFile;
            if (args.length > 1)
                originalFile = args[1] + "/" + dataSetName + ".arff";
            else
                originalFile = "/Users/fatty/Downloads/ml_datasets_arff/" + dataSetName + ".arff";
            int classIndex = dataSetClassIndex.get(dataSetName);
            Instances original = ConverterUtils.DataSource.read(originalFile);
            Helper.setDataSetClassIndex(original, dataSetClassIndex.get(dataSetName));

            // Add two noise column.
            //original = normalize(nominalToNumeric(addNoiseColumns(original, numInterruptColumns, classIndex)));

            // Define the enumerations.
            Misser misser = new KangMisser(0.0, 0.5);
            Class<?>[] classifiers = null;
            if (original.classAttribute().isNominal())
                classifiers = new Class[] {UnimputedSMO.class, IBk.class, UnimputedDecisionTable.class, UnimputedJ48.class};
            else
                classifiers = new Class[] {LinearRegression.class, IBk.class, MultilayerPerceptron.class, SMOreg.class};
            classifiers = new Class[] {UnimputedSMO.class};
            Class<?>[] imputers = new Class[]{CopyImputer.class, MEIImputer.class,
                    //HotDeckImputer.class, KNNImputer.class,
                    //LLRImputer.class,
                    FastLLRImputer.class,
                    null};
            double[] missingRatios = {0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};
            missingRatios = new double[] {0, 0.3, 0.5};
            int missRepeat = 30;
            double[][][] table = new double[classifiers.length][imputers.length][missingRatios.length];

            // Calculate the whole table here.
            for (int rpt=0; rpt<missRepeat; ++rpt) {
                System.out.println(dataSetName + ": round " + rpt + "/" + missRepeat + " starts.");
                for (int k = 0; k < missingRatios.length; ++k) {
                    Instances missed = misser.miss(original, missingRatios[k], classIndex);

                    for (int j = 0; j < imputers.length; ++j) {
                        // Check the missing data.
                        //System.out.println("Missing ratio before imputed: " + checkMissingRatio(missed));

                        Instances imputed;
                        if (imputers[j] == null) { // Complete dataset.
                            imputed = new Instances(original);
                        } else { // Imputed dataset.
                            Imputer imputer = (Imputer) imputers[j].newInstance();
                            imputed = imputer.impute(missed, classIndex);
                        }
                        //System.out.println("Missing ratio after imputed: " + checkMissingRatio(imputed));

                        for (int i=0; i < classifiers.length; ++i) {
                            Classifier classifier = (Classifier) classifiers[i].newInstance();
                            table[i][j][k] += evaluateDataSet(new Instances(imputed), classifier);
                        }
                    }
                }
                System.out.println(dataSetName + ": round " + rpt + "/" + missRepeat + " ends.");
            }

            // Display the results.
            for (int i=0; i<table.length; ++i) {
                for (int j=0; j<table[0].length; ++j) {
                    System.out.printf("%20s, ", classifiers[i].getSimpleName());
                    if (imputers[j] == null)
                        System.out.printf("%15s, ", "Complete");
                    else
                        System.out.printf("%15s, ", imputers[j].getSimpleName());
                    for (int k=0; k<table[0][0].length; ++k) {
                        if (original.classAttribute().isNominal()) // Classification.
                            table[i][j][k] = 1-table[i][j][k]/missRepeat;
                        else // Regression.
                            table[i][j][k] /= missRepeat;

                        if (j>0) // Calculate the relative improvements compared with un-imputed dataset.
                            table[i][j][k] = table[i][j][k]/table[i][0][k];

                        if (k == table[0][0].length-1)
                            System.out.printf("%7.4f\n", table[i][j][k]);
                        else
                            System.out.printf("%7.4f, ", table[i][j][k]);
                    }
                }
            }
        } catch (IllegalArgumentException | NullPointerException e) {
            System.out.println("Error occurs in main task. Details: " + e.getMessage());
        } catch (MissException e) {
            System.out.println("Miss data error. Details: " + e.getMessage());
        } catch (ImputeException e) {
            System.out.println("Impute data error. Details: " + e.getMessage());
            throw e;
        } catch (Exception e) {
            System.out.println("Unhandled error. Details: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
