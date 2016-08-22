package com.fatty.ml;

import weka.core.Instances;

/**
 * Created by fatty on 16/8/21.
 */
public interface Misser {
    void miss(String srcArffFile, String destArffFile, double ratio)
            throws IllegalArgumentException, NullPointerException, MissException;
    Instances miss(Instances data, double ratio) throws IllegalArgumentException, NullPointerException;
}
