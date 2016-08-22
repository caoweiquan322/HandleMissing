package com.fatty.ml;

import com.fatty.Helper;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import java.util.Random;

/**
 * Created by fatty on 16/8/21.
 */
public class KangMisser implements Misser {
    protected double singleInstanceMissMin;
    protected double singleInstanceMissRange;
    protected Random rnd;

    public KangMisser(double singleInstanceMissMin,
                      double singleInstanceMissMax) throws IllegalArgumentException {
        Helper.checkNotNegative("singleInstanceMissMin", singleInstanceMissMin);
        Helper.checkNotNegative("singleInstanceMissMax-singleInstanceMissMin",
                singleInstanceMissMax - singleInstanceMissMin);
        Helper.checkNotNegative("1-singleInstanceMissMax", 1.0 - singleInstanceMissMax);

        this.singleInstanceMissMin = singleInstanceMissMin;
        this.singleInstanceMissRange = singleInstanceMissMax - singleInstanceMissMin;
        this.rnd = new Random(); // A randomized random generator.
    }

    @Override
    public void miss(String srcArffFile, String destArffFile, double ratio)
            throws IllegalArgumentException, NullPointerException, MissException {
        Helper.checkFileExists(srcArffFile);
        Helper.checkNotNullNorEmpty("destArffFile", destArffFile);
        Helper.checkNotNegative("ratio", ratio);
        Helper.checkNotNegative("1-ratio", 1.0 - ratio);

        Instances data;
        // Parses the dataset.
        try {
            data = ConverterUtils.DataSource.read(srcArffFile);
        } catch (Exception e) {
            throw new MissException("Error occurs while parsing the source arff file. Details: " + e.getMessage(), e);
        }
        // Misses data.
        data = miss(data, ratio);
        Helper.checkNotNull("missed data set", data);
        // Writes the result.
        try {
            ConverterUtils.DataSink.write(destArffFile, data);
        } catch (Exception e) {
            throw new MissException("Error occurs while writing the generated arff file. Details: " + e.getMessage(), e);
        }
    }

    public Instances miss(Instances data, double ratio) throws IllegalArgumentException, NullPointerException {
        Helper.checkNotNull("data set", data);
        Helper.checkNotNegative("ratio", ratio);
        Helper.checkNotNegative("1-ratio", 1.0 - ratio);

        if (!data.isEmpty()) {
            Instance line;
            for (int i=0; i<data.numInstances(); ++i) {
                if (rnd.nextDouble() < ratio) {
                    line = data.get(i);
                    line = missInstance(line);
                    data.set(i, line);
                }
            }
        }
        return data;
    }

    protected Instance missInstance(Instance instance) {
        int numAttr = instance.numAttributes();
        int toMiss = (int) Math.round(
                (rnd.nextDouble()*singleInstanceMissRange+singleInstanceMissMin)*numAttr);
        toMiss = Math.max(toMiss, 1);
        toMiss = Math.min(toMiss, numAttr);
        int[] indices = firstNRandInt(toMiss, numAttr);
        for (int i=0; i<toMiss; ++i) {
            instance.setMissing(indices[i]);
        }

        return instance;
    }

    protected int[] firstNRandInt(int numToGenerate, int total) {
        int[] array = new int[total];
        for (int i=0; i<array.length; ++i) {
            array[i] = i;
        }
        int toSwap;
        int tmp;
        for (int i=0; i< Math.min(numToGenerate, total-1); ++i) {
            toSwap = i + rnd.nextInt(total - i);
            // Swap two elements.
            tmp = array[toSwap];
            array[toSwap] = array[i];
            array[i] = tmp;
        }
        return array;
    }
}
