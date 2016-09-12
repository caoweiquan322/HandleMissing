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
import info.debatty.java.graphs.*;
import info.debatty.java.graphs.build.NNDescent;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Created by caowq on 2016/9/12.
 */
public class UniformLLR extends AbstractClassifier {
    protected static int MIN_INSTANCES_TO_TRAIN = 3;
    protected static int MIN_ATTRIBUTE_TO_RECONSTRUCT = 2;
    protected static int DEFAULT_K = 50;
    protected Instances completeData;
    protected int initK;
    protected Graph<Instance> graph;
    protected int classIndex = -1;

    public void setSpeedup(double speedup) {
        this.speedup = speedup;
    }

    protected double speedup = 4.0;

    public void setLLRStrategy(LLRStrategy strategy) {
        this.llrStrategy = strategy;
    }

    public void setNnStrategy(NNStrategy strategy) {
        this.nnStrategy = strategy;
    }

    protected NNStrategy nnStrategy;
    protected LLRStrategy llrStrategy;

    public enum NNStrategy {
        BruteForce,
        Approximate
    }
    public enum LLRStrategy {
        Average,
        OptimizeSlow,
        Optimize2d,
        Optimize1d
    }

    public UniformLLR() {
        this.initK = DEFAULT_K;
        this.completeData = null;
        this.nnStrategy = NNStrategy.Approximate;
        this.llrStrategy = LLRStrategy.Optimize1d;
    }

    public UniformLLR(int initK) {
        this.initK = initK;
        this.completeData = null;
        this.nnStrategy = NNStrategy.Approximate;
        this.llrStrategy = LLRStrategy.Optimize1d;
    }

    public UniformLLR(int initK, NNStrategy nnStrategy, LLRStrategy llrStrategy) {
        this.initK = initK;
        this.completeData = null;
        this.nnStrategy = nnStrategy;
        this.llrStrategy = llrStrategy;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Helper.checkNotNull("instances", instances);
        if (instances.classIndex() < 0) {
            throw new Exception("The class index is not set yet.");
        }
        classIndex = instances.classIndex();

        completeData = new Instances(instances, 0);
        for (Instance instance : instances) {
            if (!instance.classIsMissing() && !instance.hasMissingValue()) {
                completeData.add(instance);
            }
        }
        if (completeData.numInstances() < MIN_INSTANCES_TO_TRAIN) {
            throw new Exception("Number of complete instances is too few to train a kNN based LLR classifier.");
        }

        // Build the k-NN graph if necessary.
        if (nnStrategy == NNStrategy.Approximate)
            buildKNNGraph();
    }

    protected void buildKNNGraph() {
        // Create the nodes
        int count = completeData.numInstances();
        ArrayList<Node> nodes = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            // The value of our nodes will be weka instance.
            nodes.add(new Node<>(String.valueOf(i), completeData.get(i)));
        }
        // Instantiate and configure the build algorithm
        NNDescent builder = new NNDescent();
        builder.setK(initK);
        // early termination coefficient
        builder.setDelta(0.1);
        // sampling coefficient
        builder.setRho(0.2);
        builder.setMaxIterations(10);
        builder.setSimilarity(new SimilarityInterface<Instance>() {
            @Override
            public double similarity(Instance v1, Instance v2) {
                // Ignore instance compatibility checking.
                double diff = 0.0;
                double a;
                for (int i=0; i<v1.numAttributes(); ++i) {
                    if (i != classIndex && !v1.isMissing(i) && !v2.isMissing(i)) {
                        a = v1.value(i)-v2.value(i);
                        diff += a * a;
                    }
                }
                return 1.0 / (1.0 + Math.sqrt(diff));
            }
        });

        // Run the algorithm and get computed graph
        graph = builder.computeGraph(nodes);
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
        if (llrStrategy == LLRStrategy.Average) { // Average strategy.
            weights = new double[k];
            for (int i=0; i<k; ++i)
                weights[i] = 1.0/k;
        } else { // Optimization strategy.
            int d = completeIndices.size();
            boolean is1DOptimize = llrStrategy == LLRStrategy.Optimize1d;
            double[][] halfP = is1DOptimize ? new double[d+1][k] : new double[d][k];
            for (int j = 0; j < k; ++j) {
                Instance nb = completeData.get(nearestIndices[j]);
                for (int i = 0; i < d; ++i) {
                    halfP[i][j] = nb.value(completeIndices.get(i));
                }
                if (is1DOptimize) halfP[d][j] = 1.0;
            }
            double[] halfQ = is1DOptimize ? new double[d+1] : new double[d];
            for (int i = 0; i < d; ++i) {
                halfQ[i] = instance.value(completeIndices.get(i));
            }
            if (is1DOptimize) halfQ[d] = 1.0;

            if (llrStrategy == LLRStrategy.Optimize1d)
                weights = smoBased1dLLRSolver(halfP, halfQ);
            else if (llrStrategy == LLRStrategy.Optimize2d)
                weights = smoBased2dLLRSolver(halfP, halfQ);
            else
                weights = slowSolver(halfP, halfQ);
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
        if (nnStrategy == NNStrategy.BruteForce) {
            // Calculate distances.
            int numInstances = completeData.numInstances();
            double[] distances = new double[numInstances];
            for (int i = 0; i < numInstances; ++i) {
                distances[i] = distance(completeData.get(i), instance, completeIndices);
            }
            // Sort top-K nearest neighbours within O(N) time efficiently.
            return Helper.getLeastIndices(distances, k);
        } else {
            // Approximate version.
            NeighborList nl = graph.fastSearch(instance, k, speedup);
            if (nl.size() != k) { // Adapt k if necessary.
                if (nl.size() > 0)
                    k = nl.size();
                else
                    throw new Exception("Expected get " + k + " neighbors but got " + nl.size());
            }

            int[] indices = new int[k];
            int i=0;
            Neighbor nb;
            Iterator<Neighbor> itr = nl.iterator();
            while (itr.hasNext()) {
                nb = itr.next();
                indices[i] = Integer.parseInt(nb.node.id);
                ++i;
            }
            return indices;
        }
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

    public static double[] smoBased1dLLRSolver(double[][] A, double[] b) {
        Helper.checkIntEqual(A.length, b.length);
        int k = A[0].length;
        double EPSILON = 1e-4;
        int MAX_ITR = 100;

        // Initialization.
        double[] w = new double[k];
        for (int i=0; i<k; ++i) w[i] = 1.0/k;
        double[][] At = matTranspose(A);
        double[][] AtA = matMultiply(At, A);
        double[] q = diagonal(AtA);
        double[] c = vecSub(matMultiply(A, w), b);
        double[] alpha = matMultiply(At, c);
        double[] cnst = null;// Need be initialized in each iteration.

        // Iterations.
//        double sum0 = 0.0;
//        for (double x: vecSub(matMultiply(A, w), b)) {
//            //System.out.printf("%5.3f,", x);
//            sum0 += Math.abs(x);
//        }
//        System.out.printf("Itr: %04d, %8.6f\n", 0, sum0);

        double delta = 0.0, nwi;
        for (int itr=0; itr<MAX_ITR; ++itr) {
            boolean hasBreak = false;
            for (int i = 0; i < k; ++i) {
                if (Math.abs(alpha[i]) > EPSILON && Math.abs(w[i]) > EPSILON) {
                    hasBreak = true;
                    cnst = vecSub(c, At[i], w[i], cnst); // Update the const part of the 1-d optimization problem.
                    nwi = -dotProd(At[i], cnst)/q[i]; // Update the solution vector.
                    if (nwi<0) nwi=0;
                    //if (nwi>1) nwi=1.0;
                    delta = nwi - w[i];
                    w[i] = nwi;
                    c = vecAdd(c, At[i], delta, c);
                    alpha = vecAdd(alpha, AtA[i], delta, alpha);
                }
            }

            // Display the partial results.
//            double[] diff = vecSub(matMultiply(A, w), b);
//            double sum = 0.0;
//            for (double x: diff) {
//                //System.out.printf("%5.3f,", x);
//                sum += Math.abs(x);
//            }
//            System.out.printf("Itr: %04d, %8.6f\n", itr, sum);

            // Optimized.
            if (!hasBreak)
                break;
        }

        return w;
    }

    public static double[] smoBased2dLLRSolver(double[][] A, double[] b) {
        Helper.checkIntEqual(A.length, b.length);
        int k = A[0].length;
        if (k<2) {
            throw new IllegalArgumentException("The k of QP problem must be at least 2.");
        }
        double EPSILON = 1e-4;
        int MAX_ITR = 100;

        // Initialization.
        double[] w = new double[k];
        for (int i=0; i<k; ++i) w[i] = 1.0/k;
        double[][] At = matTranspose(A);
        double[][] AtA = matMultiply(At, A);
        double[] q = diagonal(AtA);
        double[] c = vecSub(matMultiply(A, w), b);
        double[] alpha = matMultiply(At, c);
        double[] cnst = null;// Need be initialized in each iteration.

        // Iterations.
//        double[] diff = vecSub(matMultiply(A, w), b);
//        double sum1 = 0.0, sum2=0;
//        for (int i=0; i<k; ++i) sum1 += w[i];
//        for (int i=0; i<b.length; ++i) sum2 += diff[i]*diff[i];
//        System.out.printf("Itr: %04d, %8.6f, %8.6f\n", 0, Math.sqrt(sum2), sum1);

        double beta = 0.0, nwi, nwj, bound;
        double[] Ai_j = null;
        double nzSum = 1.0, nzCount = k;
        Random r = new Random();
        for (int itr=0; itr<MAX_ITR; ++itr) {
            boolean hasBreak = false;
            for (int i = 0; i < k; ++i) {
                if (Math.abs(alpha[i]) > EPSILON && Math.abs(w[i]) > EPSILON) {
                    int j = i;
                    while (j==i) j = r.nextInt(k);
                    hasBreak = true;

                    bound = w[i]+w[j];
                    Ai_j = vecSub(At[i], At[j], Ai_j);
                    cnst = vecSub(c, Ai_j, w[i], cnst); // Update the const part of the 2-d optimization problem.

                    nwi = -dotProd(Ai_j, cnst)/dotProd(Ai_j, Ai_j); // Update the solution vector.
                    if (nwi<0) nwi=0;
                    if (nwi>bound) nwi=bound;
                    nwj = bound-nwi;
                    if (Math.abs(w[i]) > EPSILON) {
                        --nzCount;
                        nzSum -= w[i];
                    }
                    if (Math.abs(w[j]) > EPSILON) {
                        --nzCount;
                        nzSum -= w[j];
                    }
                    if (Math.abs(nwi) > EPSILON) {
                        ++nzCount;
                        nzSum += nwi;
                    }
                    if (Math.abs(nwj) > EPSILON) {
                        ++nzCount;
                        nzSum += nwj;
                    }
                    if (nzCount-EPSILON < 0) { // The problem has been optimized.
                        hasBreak = false;
                        break;
                    }

                    c = vecAdd(c, Ai_j, nwi-w[i], c);
                    alpha = vecAdd(alpha, AtA[i], nwi-w[i], alpha);
                    alpha = vecSub(alpha, AtA[j], nwj-w[j], alpha);
                    for (int m=0; m<k; ++m)
                        alpha[m] -= (nzSum/nzCount-beta);

                    w[i] = nwi;
                    w[j] = nwj;
                    beta = nzSum/nzCount;
                }
            }

            // Display the partial results.
//            double[] diff0 = vecSub(matMultiply(A, w), b);
//            double sum10 = 0.0, sum20=0;
//            for (int i=0; i<k; ++i) sum10 += w[i];
//            for (int i=0; i<b.length; ++i) sum20 += diff0[i]*diff0[i];
//            System.out.printf("Itr: %04d, %8.6f, %8.6f\n", itr, Math.sqrt(sum20), sum10);

            // Optimized.
            if (!hasBreak)
                break;
        }

        return w;
    }

    public static double[] slowSolver(double[][] A, double[] b) {
        int k = A[0].length;
        int d = A.length;
        double[][] At = matTranspose(A);
        double[][] P = matMultiply(At, A);
        double[] Q = matMultiply(At, b);
        for (int i=0; i<d; ++i)Q[i] = -Q[i]; // Since we are optimizing |Ax-b|
        double[] initW = new double[k];
        for (int i=0; i<k; ++i) initW[i] = 1.0/k;

        OptimizationRequest or = new OptimizationRequest();
        PDQuadraticMultivariateRealFunction objective =
                new PDQuadraticMultivariateRealFunction(P, Q, 0.0);
        or.setF0(objective);
        ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[k];
        double[][] identity = DoubleFactory2D.dense.diagonal(DoubleFactory1D.dense.make(k, -1.0)).toArray();
        for (int i = 0; i < k; ++i)
            inequalities[i] = new LinearMultivariateRealFunction(identity[i], 0.0);
        or.setFi(inequalities);
        double[][] eqA = new double[1][k];
        double[] eqB = new double[] {1.0};
        for (int i=0; i<k; ++i) eqA[0][i] = 1.0;
        or.setA(eqA);
        or.setB(eqB);
        or.setInitialPoint(initW);
        JOptimizer opt = new JOptimizer();
        opt.setOptimizationRequest(or);
        try {
            int retCode = opt.optimize();
        } catch (Exception e) {
            System.out.println("Error occurs while optimizing. Details: " + e.getMessage());
            return initW;
        }
        return opt.getOptimizationResponse().getSolution();
    }


    public static double[] diagonal(double[][] A) {
        Helper.checkIntEqual(A.length, A[0].length);
        int n = A.length;
        double[] d = new double[n];
        for (int i=0; i<n; ++i)
            d[i] = A[i][i];
        return d;
    }

    public static double[] vecSub(double[] a, double[] b) {
        return vecSub(a, b, 1.0, null);
    }

    public static double[] vecSub(double[] a, double[] b, double[] c) {
        return vecSub(a, b, 1.0, c);
    }

    public static double[] vecSub(double[] a, double[] b, double bScale, double[] c) {
        Helper.checkIntEqual(a.length, b.length);
        int n = a.length;
        if (c==null)
            c = new double[n];
        for (int i=0; i<n; ++i)
            c[i] = a[i]-b[i]*bScale;
        return c;
    }

    public static double[] vecAdd(double[] a, double[] b) {
        return vecAdd(a, b, 1.0, null);
    }

    public static double[] vecAdd(double[] a, double[] b, double[] c) {
        return vecAdd(a, b, 1.0, c);
    }

    public static double[] vecAdd(double[] a, double[] b, double bScale, double[] c) {
        return vecSub(a, b, -bScale, c);
    }

    public static double dotProd(double[] a, double[] b) {
        Helper.checkIntEqual(a.length, b.length);
        double prod = 0.0;
        for (int i=0; i<a.length; ++i)
            prod += a[i]*b[i];
        return prod;
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
}

