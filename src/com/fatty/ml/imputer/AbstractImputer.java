package com.fatty.ml.imputer;

import com.fatty.Helper;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Created by fatty on 16-8-23.
 */
public class AbstractImputer implements Imputer {
    @Override
    public void impute(String srcArffFile, String destArffFile, int classIndex)
            throws IllegalArgumentException, NullPointerException, ImputeException {
        Helper.checkFileExists(srcArffFile);
        Helper.checkNotNullNorEmpty("destArffFile", destArffFile);

        Instances data;
        // Parses the dataset.
        try {
            data = ConverterUtils.DataSource.read(srcArffFile);
        } catch (Exception e) {
            throw new ImputeException("Error occurs while parsing the source arff file. Details: " + e.getMessage(), e);
        }
        // Misses data.
        data = impute(data, classIndex);
        Helper.checkNotNull("missed data set", data);
        // Writes the result.
        try {
            ConverterUtils.DataSink.write(destArffFile, data);
        } catch (Exception e) {
            throw new ImputeException("Error occurs while writing the generated arff file. Details: " + e.getMessage(), e);
        }
    }

    @Override
    public Instances impute(Instances instances, int classIndex) throws ImputeException {
        throw new ImputeException("The impute function is not implemented in abstract class.");
    }
}
