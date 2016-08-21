package com.fatty.ml;

import com.fatty.Helper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
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

        List<String> titles = new ArrayList<>();
        List<String> contents = new ArrayList<>();
        // Read source file.
        try (FileReader fr = new FileReader(srcArffFile);
             BufferedReader br = new BufferedReader(fr)) {
            boolean hasReadTitles = false;
            String line;
            while((line = br.readLine()) != null) {
                if (hasReadTitles) {
                    contents.add(line.trim());
                } else {
                    titles.add(line.trim());
                }
                // Update the state variable.
                if (line.toUpperCase().startsWith("@DATA")) {
                    hasReadTitles = true;
                }
            }
        } catch (Exception e) {
            throw new MissException("Error parsing source ARFF file: " + srcArffFile, e);
        }

        // Do missing.
        if (!contents.isEmpty()) {
            for (int i=0; i<contents.size(); ++i) {
                if (rnd.nextFloat() < ratio && !contents.get(i).isEmpty()) {
                    try {
                        contents.set(i, missInstance(contents.get(i)));
                    } catch (Exception e) {
                        // Yield.
                        System.out.println("Error occurs while missing one instance. Details: " + e.getMessage());
                    }
                }
            }
        }

        // Write destination file.
        try (FileWriter fw = new FileWriter(destArffFile);
             BufferedWriter bw = new BufferedWriter(fw)) {
            for (String s : titles) {
                bw.write(s);
                bw.write("\n");
            }
            for (String s : contents) {
                bw.write(s);
                bw.write("\n");
            }
        } catch (Exception e) {
            throw new MissException("Error writing destination ARFF file. Details: ", e);
        }
    }

    public String missInstance(String instance) throws IllegalArgumentException {
        // Ignore parameter checking.
        String[] parts = instance.split(",");
        if (parts.length == 0) {
            throw new IllegalArgumentException("Expected instance contains comma, but got " + instance);
        }

        int toMiss = (int) Math.round(
                (rnd.nextDouble()*singleInstanceMissRange+singleInstanceMissMin)*parts.length);
        toMiss = Math.max(toMiss, 1);
        toMiss = Math.min(toMiss, parts.length);
        int[] indices = firstNRandInt(toMiss, parts.length);
        for (int i=0; i<toMiss; ++i) {
            parts[indices[i]] = "?";
        }

        return Helper.join(",", parts);
    }

    protected int[] firstNRandInt(int numToGenerate, int total) {
        int[] array = new int[total];
        for (int i=0; i<array.length; ++i) {
            array[i] = i;
        }
        int toSwap = -1;
        int tmp;
        for (int i=0; i< Math.min(numToGenerate, total-1); ++i) {
            toSwap = i + rnd.nextInt(total - 1 - i);
            // Swap two elements.
            tmp = array[toSwap];
            array[toSwap] = array[i];
            array[i] = tmp;
        }
        return array;
    }
}
