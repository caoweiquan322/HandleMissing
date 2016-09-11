package com.fatty.ml.imputer;

import com.fatty.Helper;
import weka.core.Instances;

/**
 * Created by fatty on 16-8-30.
 */
public class CopyImputer extends AbstractImputer {
    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        Helper.checkNotNull("instances", instances);
        // Do imputing.
        try {
            Helper.setDataSetClassIndex(instances, classIndex);
            return new Instances(instances);
        } catch (Exception e) {
            throw new ImputeException("Error occurs while imputing data set. Details: " + e.getMessage(), e);
        }
    }
}
