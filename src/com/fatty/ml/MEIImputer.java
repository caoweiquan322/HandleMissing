package com.fatty.ml;

import com.fatty.Helper;

/**
 * Created by fatty on 16/8/21.
 */
public class MEIImputer implements Imputer {
    @Override
    public void impute(String srcArffFile, String destArffFile) {
        Helper.checkFileExists(srcArffFile);
        Helper.checkNotNullNorEmpty("destArffFile", destArffFile);
    }
}
