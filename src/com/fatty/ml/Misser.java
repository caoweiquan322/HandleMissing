package com.fatty.ml;

/**
 * Created by fatty on 16/8/21.
 */
public interface Misser {
    void miss(String srcArffFile, String destArffFile, double ratio);
}
