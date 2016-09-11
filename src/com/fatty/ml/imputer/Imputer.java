package com.fatty.ml.imputer;

import weka.core.Instances;

/**
 * Created by fatty on 16/8/21.
 */
public interface Imputer {
    void impute(String srcArffFile, String destArffFile, int classIndex)
            throws IllegalArgumentException, NullPointerException, ImputeException;

    Instances impute(Instances instances, int classIndex) throws ImputeException;
}
