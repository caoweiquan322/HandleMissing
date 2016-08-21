package com.fatty.ml;

import com.fatty.Helper;

/**
 * Created by fatty on 16/8/21.
 */
public class KangMisser implements Misser {
    @Override
    public void miss(String srcArffFile, String destArffFile, double ratio) {
        Helper.checkFileExists(srcArffFile);
        Helper.checkNotNullNorEmpty("destArffFile", destArffFile);
        Helper.checkNotNegative("ratio", ratio);
    }
}
