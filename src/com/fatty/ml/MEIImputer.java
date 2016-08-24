package com.fatty.ml;

import com.fatty.Helper;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Created by fatty on 16/8/21.
 */
public class MEIImputer extends AbstractImputer {
    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        Helper.checkNotNull("instances", instances);
        // Do imputing.
        try {
            Helper.setDataSetClassIndex(instances, classIndex);
            ReplaceMissingValues filter = new ReplaceMissingValues();
            filter.setInputFormat(instances);
            return Filter.useFilter(instances, filter);
        } catch (Exception e) {
            throw new ImputeException("Error occurs while imputing data set. Details: " + e.getMessage(), e);
        }
    }
}
