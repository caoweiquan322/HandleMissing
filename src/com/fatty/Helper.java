package com.fatty;

import weka.core.Instances;

import java.io.File;

/**
 * Created by fatty on 16/8/21.
 */
public class Helper {
    /**
     * This field represents a minimal positive number.
     */
    public static final double EPSILON = 1e-8;

    /**
     * Checks if two integers equal.
     * @param a the first integer.
     * @param b the second integer.
     * @throws IllegalArgumentException if a does not equal to b.
     */
    public static void checkIntEqual(int a, int b) throws IllegalArgumentException {
        if (a != b) {
            throw new IllegalArgumentException("Expected " + a + " equals to " + b);
        }
    }

    /**
     * Checks if the specified val is null.
     * @param name is the name of the value to check.
     * @param val is the value to check.
     * @param <T> could be any type extends Object.
     * @throws NullPointerException if the input val is null.
     */
    public static <T> void checkNotNull(String name, T val) throws NullPointerException {
        if (val == null) {
            throw new NullPointerException("Expected " + name + " to be not null.");
        }
    }

    public static void checkNotNullNorEmpty(String name, String val)
            throws IllegalArgumentException, NullPointerException {
        checkNotNull(name, val);
        if (val.trim().isEmpty()) {
            throw new IllegalArgumentException("Expected " + name + " to be not empty.");
        }
    }

    public static void checkFileExists(String filePath)
            throws IllegalArgumentException, NullPointerException {
        checkNotNullNorEmpty("filePath", filePath);
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IllegalArgumentException("Expected file " + filePath + " exists.");
        }
    }

    public static <T extends Number> void checkNotNegative(String name, T val) throws IllegalArgumentException {
        if (val.doubleValue() < -EPSILON) {
            throw new IllegalArgumentException("Expected " + name + " to be non-negative.");
        }
    }

    public static String join(String join, String[] strAry){
        StringBuffer sb = new StringBuffer();
        for(int i=0; i<strAry.length; i++){
            if(i==(strAry.length-1)){
                sb.append(strAry[i]);
            } else {
                sb.append(strAry[i]).append(join);
            }
        }

        return new String(sb);
    }

    public static void setDataSetClassIndex(Instances instances, int classIndex) throws IllegalArgumentException {
        Helper.checkNotNull("instances", instances);
        if (classIndex>=0 && classIndex < instances.numAttributes()) {
            instances.setClassIndex(classIndex);
        } else if (classIndex == -1) {
            instances.setClassIndex(instances.numAttributes()-1);
        } else {
            throw new IllegalArgumentException("Expected class index to be -1 or within range [0, "
                    + instances.numAttributes() + "), but got " + classIndex);
        }
    }
}
