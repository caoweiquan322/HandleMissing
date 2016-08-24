package com.fatty.ml;

import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Created by fatty on 16-8-24.
 */
public class FastLLR extends IBk {
    /** for serialization */
    private static final long serialVersionUID = 6502780192411755343L;

    /**
     * IBk classifier. Simple instance-based learner that uses the class
     * of the nearest k training instances for the class of the test
     * instances.
     *
     * @param k the number of nearest neighbors to use for prediction
     */
    public FastLLR(int k) {
        super(k);
    }

    /**
     * IB1 classifer. Instance-based learner. Predicts the class of the
     * single nearest training instance for each test instance.
     */
    public FastLLR() {
        super();
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if an error occurred during the prediction
     */
    public double [] distributionForInstance(Instance instance) throws Exception {
        if (m_Train.numInstances() == 0) {
            //throw new Exception("No training instances!");
            return m_defaultModel.distributionForInstance(instance);
        }
        if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
            m_kNNValid = false;
            boolean deletedInstance=false;
            while (m_Train.numInstances() > m_WindowSize) {
                m_Train.delete(0);
            }
            //rebuild datastructure KDTree currently can't delete
            if(deletedInstance==true)
                m_NNSearch.setInstances(m_Train);
        }

        // Select k by cross validation
        if (!m_kNNValid && (m_CrossValidate) && (m_kNNUpper >= 1)) {
            crossValidate();
        }

        m_NNSearch.addInstanceInfo(instance);

        Instances neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
        double [] distances = m_NNSearch.getDistances();
        double [] distribution = makeDistribution( neighbours, distances );

        return distribution;
    }

    /**
     * Turn the list of nearest neighbors into a probability distribution.
     *
     * @param neighbours the list of nearest neighboring instances
     * @param distances the distances of the neighbors
     * @return the probability distribution
     * @throws Exception if computation goes wrong or has no class attribute
     */
    protected double [] makeDistribution(Instances neighbours, double[] distances)
            throws Exception {

        double total = 0, weight;
        double [] distribution = new double [m_NumClasses];

        // Set up a correction to the estimator
        if (m_ClassType == Attribute.NOMINAL) {
            for(int i = 0; i < m_NumClasses; i++) {
                distribution[i] = 1.0 / Math.max(1,m_Train.numInstances());
            }
            total = (double)m_NumClasses / Math.max(1,m_Train.numInstances());
        }

        for(int i=0; i < neighbours.numInstances(); i++) {
            // Collect class counts
            Instance current = neighbours.instance(i);
            distances[i] = distances[i]*distances[i];
            distances[i] = Math.sqrt(distances[i]/m_NumAttributesUsed);
            switch (m_DistanceWeighting) {
                case WEIGHT_INVERSE:
                    weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
                    break;
                case WEIGHT_SIMILARITY:
                    weight = 1.0 - distances[i];
                    break;
                default:                                 // WEIGHT_NONE:
                    weight = 1.0;
                    break;
            }
            weight *= current.weight();
            try {
                switch (m_ClassType) {
                    case Attribute.NOMINAL:
                        distribution[(int)current.classValue()] += weight;
                        break;
                    case Attribute.NUMERIC:
                        distribution[0] += current.classValue() * weight;
                        break;
                }
            } catch (Exception ex) {
                throw new Error("Data has no class attribute!");
            }
            total += weight;
        }

        // Normalise distribution
        if (total > 0) {
            Utils.normalize(distribution, total);
        }
        return distribution;
    }
}
