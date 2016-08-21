package com.fatty.ml;

/**
 * Created by fatty on 16/8/21.
 */
public interface Imputer {
    void impute(String srcArffFile, String destArffFile);
}
