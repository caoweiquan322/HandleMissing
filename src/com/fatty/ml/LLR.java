package com.fatty.ml;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import com.fatty.Helper;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.optimizers.JOptimizer;
import com.joptimizer.optimizers.OptimizationRequest;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

/**
 * Created by fatty on 16-8-23.
 */
public class LLR extends AbstractClassifier {
    protected static int MIN_INSTANCES_TO_TRAIN = 3;
    protected static int MIN_ATTRIBUTE_TO_RECONSTRUCT = 2;
    protected static int DEFAULT_K = 50;
    protected Instances completeData;
    protected int initK;

    public void setStrategy(LLRStrategy strategy) {
        this.strategy = strategy;
    }

    protected LLRStrategy strategy;

    public enum LLRStrategy {
        Average,
        Optimize
    }

    public LLR() {
        this.initK = DEFAULT_K;
        this.completeData = null;
        this.strategy = LLRStrategy.Optimize;
    }

    public LLR(int initK) {
        this.initK = initK;
        this.completeData = null;
        this.strategy = LLRStrategy.Optimize;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Helper.checkNotNull("instances", instances);
        if (instances.classIndex() < 0) {
            throw new Exception("The class index is not set yet.");
        }

        completeData = new Instances(instances, 0);
        for (Instance instance : instances) {
            if (!instance.classIsMissing() && !instance.hasMissingValue()) {
                completeData.add(instance);
            }
        }
        if (completeData.numInstances() < MIN_INSTANCES_TO_TRAIN) {
            throw new Exception("Number of complete instances is too few to train a LLR classifier.");
        }
    }

    @Override
    /**
     * Classify an instance. Note that this operation will fill in the missing fields of instance.
     */
    public double classifyInstance(Instance instance) throws Exception {
        if (!instance.classIsMissing()) {
            throw new Exception("The instance to classify is not missing class value.");
        }
        double[] hist = distributionForInstance(instance);

        if (instance.classAttribute().isNominal()) {
            return Utils.maxIndex(hist);
        } else {
            return hist[0];
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Helper.checkNotNull("Trained complete data", completeData);
        Helper.checkPositive("Trained instances number", completeData.numInstances());
        if (!instance.equalHeaders(completeData.get(0))) {
            throw new Exception("The instance headers do not match.");
        }

        int numInstances = completeData.numInstances();
        int k = Math.min(initK, numInstances);
        int classIndex = completeData.classIndex();

        // Split attributes into complete/incomplete sets.
        List<Integer> completeIndices = new ArrayList<>();
        List<Integer> incompleteIndices = new ArrayList<>();
        for (int i=0; i<instance.numAttributes(); ++i) {
            if (instance.isMissing(i)) {
                incompleteIndices.add(i);
            } else if (instance.attribute(i).isNumeric() && i != classIndex) {
                completeIndices.add(i);
            }
        }
        if (completeIndices.size() < MIN_ATTRIBUTE_TO_RECONSTRUCT) {
            // Failed to classify.
            throw new Exception("The number of complete attributes is too few to reconstruct. At least "
                    + MIN_ATTRIBUTE_TO_RECONSTRUCT + " is required.");
        }

        // Get indices of k-NN.
        int[] nearestIndices = calculateKNN(instance, completeIndices, k);
        // In case there are not exactly k neighbors returned.
        if (k != nearestIndices.length)
            k = nearestIndices.length;

        // Solve the QP for the best reconstruction.
        double[] weights;
        if (strategy == LLRStrategy.Average) { // Average strategy.
            weights = new double[k];
            for (int i=0; i<k; ++i)
                weights[i] = 1.0/k;
        } else { // Optimization strategy.

            double[][] halfP = new double[completeIndices.size()][k];
            for (int j = 0; j < k; ++j) {
                Instance nb = completeData.get(nearestIndices[j]);
                for (int i = 0; i < completeIndices.size(); ++i) {
                    halfP[i][j] = nb.value(completeIndices.get(i));
                }
            }
            double[] halfQ = new double[completeIndices.size()];
            for (int i = 0; i < completeIndices.size(); ++i) {
                halfQ[i] = -instance.value(completeIndices.get(i));
            }
            double[] nY = new double[k];
            for (int i = 0; i < k; ++i) {
                nY[i] = -completeData.get(nearestIndices[i]).classValue();
            }

            DoubleMatrix2D A = DoubleFactory2D.dense.make(1, k, 1.0);
            DoubleMatrix1D B = DoubleFactory1D.dense.make(1, 1.0);
            ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[k];
            double[][] identity = DoubleFactory2D.dense.diagonal(DoubleFactory1D.dense.make(k, -1.0)).toArray();
            for (int i = 0; i < k; ++i)
                inequalities[i] = new LinearMultivariateRealFunction(identity[i], 0.0);

            weights = emIterateForOptimalWeights(halfP, halfQ, nY, inequalities, A, B, k, completeIndices.size());
            Helper.checkNotNull("weights", weights);
            Helper.checkIntEqual(weights.length, k);
        }

        // Output the imputed data.
        double[] classHist = null;
        for (int i: incompleteIndices) {
            Attribute attr = instance.attribute(i);
            if (attr.isNumeric()) {
                double sum = 0.0;
                for (int j=0; j<k; ++j) {
                    sum += weights[j]*completeData.get(nearestIndices[j]).value(i);
                }
                if (instance.classIndex() == i) {
                    classHist = new double[] {sum};
                } else {
                    instance.setValue(i, sum);
                }
            } else if (attr.isNominal()) {
                Helper.checkPositive("attribute " + i + " values number", attr.numValues());
                double[] hist = new double[attr.numValues()];
                for (int j=0; j<k; ++j) {
                    hist[(int)Math.round(completeData.get(nearestIndices[j]).value(i))] += weights[j];
                }

                if (instance.classIndex() == i) {
                    classHist = hist;
                } else {
                    instance.setValue(i, Utils.maxIndex(hist));
                }
            } else {
                throw new Exception("We could only handle numeric/nominal attributes by now.");
            }
        }

        return classHist;
    }

    protected int[] calculateKNN(Instance instance, List<Integer> completeIndices, int k) throws Exception {
        // Calculate distances.
        int numInstances = completeData.numInstances();
        double[] distances = new double[numInstances];
        for (int i=0; i<numInstances; ++i) {
            distances[i] = distance(completeData.get(i), instance, completeIndices);
        }

        // Sort top-K nearest neighbours within O(N) time efficiently.
        return Helper.getLeastIndices(distances, k);
    }

    protected double[] emIterateForOptimalWeights(double[][] halfP, double[] halfQ, double[] nY,
                                                  ConvexMultivariateRealFunction[] inequalities,
                                                  DoubleMatrix2D A, DoubleMatrix1D B,
                                                  int k, int numCompleteAttributes) throws Exception {
        double[] instanceWeights = new double[k];
        for (int i=0; i<k; ++i)
            instanceWeights[i] = 1.0/k;

        double[] attributeWeights = new double[numCompleteAttributes];
        for (int i=0; i<numCompleteAttributes; ++i)
            attributeWeights[i] = 1.0;
        double[][] halfPt = matTranspose(halfP);

        try {
            instanceWeights = solveOptimalInstanceWeights(halfP, halfPt, halfQ, inequalities, A, B, instanceWeights, attributeWeights);
        } catch (Exception e) {
            System.out.println("Error solve instance. Details: " + e.getMessage());
        }
        return instanceWeights;
    }

    protected double[] solveOptimalInstanceWeights(double[][] halfP, double[][] halfPt, double[] halfQ,
                                                   ConvexMultivariateRealFunction[] inequalities,
                                                   DoubleMatrix2D A, DoubleMatrix1D B,
                                                   double[] initInstanceWeights,
                                                   double[] attributeWeights) throws Exception {
        double[][] nHalfPt = new double[halfPt.length][halfPt[0].length];
        for (int j=0; j<halfPt[0].length; ++j) {
            double w = attributeWeights[j] * attributeWeights[j];
            for (int i = 0; i < halfPt.length; ++i)
                nHalfPt[i][j] = halfPt[i][j] * w;
        }
        double[][] P = matMultiply(nHalfPt, halfP);
        double[] Q = matMultiply(nHalfPt, halfQ);

        OptimizationRequest or = new OptimizationRequest();
        PDQuadraticMultivariateRealFunction objective =
                new PDQuadraticMultivariateRealFunction(P, Q, 0.0);
        or.setF0(objective);
        or.setFi(inequalities);
        or.setA(A);
        or.setB(B);
        or.setInitialPoint(initInstanceWeights);
        //or.setTolerance(1e-3);
        JOptimizer opt = new JOptimizer();
        opt.setOptimizationRequest(or);
        int retCode = opt.optimize();
        return opt.getOptimizationResponse().getSolution();
    }

    protected double[] solveOptimalAttributeWeights(double[][] halfP, double[][] halfPt, double[] nY,
                                                    double[] initAttributeWeights, double[] instanceWeights) throws Exception {
        double[][] nHalfP = new double[halfP.length][halfP[0].length];
        for (int j=0; j<halfP[0].length; ++j) {
            double w = 1.0;//instanceWeights[j] * instanceWeights[j];
            for (int i = 0; i < halfP.length; ++i)
                nHalfP[i][j] = halfP[i][j] * w;
        }
        double[][] P = matMultiply(nHalfP, halfPt);
        double[] Q = matMultiply(nHalfP, nY);
        DoubleMatrix2D matP = DoubleFactory2D.dense.make(P);
        DoubleMatrix2D matQ = DoubleFactory2D.dense.make(new double[][]{Q});
        matQ = Algebra.DEFAULT.transpose(matQ);

        DoubleMatrix2D ans = Algebra.DEFAULT.solve(matP, matQ);
        return Algebra.DEFAULT.transpose(ans).toArray()[0];

//        OptimizationRequest or = new OptimizationRequest();
//        PDQuadraticMultivariateRealFunction objective =
//                new PDQuadraticMultivariateRealFunction(P, Q, 0.0);
//        or.setF0(objective);
//        //or.setInitialPoint(initAttributeWeights);
//        // or.setTolerance(1e-3);
//        JOptimizer opt = new JOptimizer();
//        opt.setOptimizationRequest(or);
//        int retCode = opt.optimize();
//        return opt.getOptimizationResponse().getSolution();
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(MIN_INSTANCES_TO_TRAIN);

        return result;
    }

    protected static double distance(Instance a, Instance b, List<Integer> caredIndices) {
        double d = 0;
        double va, vb;
        if (caredIndices == null) {
            for (int i=0; i<a.numAttributes(); ++i) {
                va = a.value(i);
                vb = b.value(i);
                d += Math.sqrt((va-vb)*(va-vb));
            }
        } else {
            for (int i: caredIndices) {
                va = a.value(i);
                vb = b.value(i);
                d += Math.sqrt((va-vb)*(va-vb));
            }
        }
        return d;
    }

    public static double[][] matTranspose(double[][] m) {
        Helper.checkNotNull("m", m);
        Helper.checkPositive("rows", m.length);
        Helper.checkPositive("columns", m[0].length);
        int rows = m.length;
        int columns = m[0].length;
        double[][] mt = new double[columns][rows];
        for (int i=0; i<rows; ++i)
            for (int j=0; j<columns; ++j)
                mt[j][i] = m[i][j];
        return mt;
    }

    public static double[][] matMultiply(double[][] a, double[][] b) {
        Helper.checkNotNull("a", a);
        Helper.checkPositive("a.rows", a.length);
        Helper.checkPositive("a.columns", a[0].length);
        Helper.checkNotNull("b", b);
        Helper.checkPositive("b.rows", b.length);
        Helper.checkPositive("b.columns", b[0].length);

        int ra = a.length, ca = a[0].length;
        int rb = b.length, cb = b[0].length;
        Helper.checkIntEqual(ca, rb);
        double[][] prod = new double[ra][cb];
        for(int i=0; i<ra; ++i) {
            for (int j=0; j<cb; ++j) {
                double sum = 0.0;
                for (int k=0; k<ca; ++k) {
                    sum += a[i][k]*b[k][j];
                }
                prod[i][j] = sum;
            }
        }
        return prod;
    }

    public static double[] matMultiply(double[][] a, double[] b) {
        Helper.checkNotNull("b", b);
        Helper.checkPositive("b.size", b.length);

        double[][] b2 = new double[b.length][1];
        for(int i=0; i<b.length; ++i)
            b2[i][0] = b[i];

        double[][] prod2 = matMultiply(a, b2);
        double[] prod = new double[prod2.length];
        for (int i=0; i<prod2.length; ++i)
            prod[i] = prod2[i][0];
        return prod;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        return null;
    }
}
