package com.fatty.ml;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.optimizers.JOptimizer;
import com.joptimizer.optimizers.OptimizationRequest;

import java.util.Random;

/**
 * Created by fatty on 16/9/11.
 */
public class FastQPSolverDemo {
    public static void main(String[] args) {
        // Data preparation.
        int k = 50;
        int d = 10;
        Random r = new Random();
        double[][] A = new double[d][k];
        double[] b = new double[d];
        double[][] Acap = new double[d+1][k];
        double[] bcap = new double[d+1];
        for (int i=0; i<d; ++i) {
            for (int j=0; j<k; ++j) {
                A[i][j] = r.nextDouble();
                Acap[i][j] = A[i][j];
            }
            b[i] = r.nextDouble();
            bcap[i] = b[i];
        }
        for (int j=0; j<k; ++j) {
            Acap[d][j] = 1.0;
        }
        bcap[d] = 1.0;
        int repeat = 10;

        // Solution A.
        double[] wa = null;
        double ta = System.nanoTime();
        for (int i=0; i<repeat; ++i)
            wa = FastLLR.smoBasedLLRSolver(Acap, bcap);
        ta = System.nanoTime()-ta;
        displayErr(A, b, wa);

        // Solution B. Compare with slow solver.
        double[] wb = null;
        double tb = System.nanoTime();
        for (int i=0; i<repeat; ++i)
            wb = slowSolver(A, b);
        tb = System.nanoTime()-tb;
        displayErr(A, b, wb);

        System.out.printf("%.3f, %.3f\n", ta/1.0/repeat/1e6, tb/1.0/repeat/1e6);
    }

    public static double[] slowSolver(double[][] A, double[] b) {
        int k = A[0].length;
        int d = A.length;
        double[][] At = FastLLR.matTranspose(A);
        double[][] P = FastLLR.matMultiply(At, A);
        double[] Q = FastLLR.matMultiply(At, b);
        for (int i=0; i<d; ++i)Q[i] = -Q[i];
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

    public static void displayErr(double[][] A, double[] b, double[] wb) {
        double[] diff = FastLLR.vecSub(FastLLR.matMultiply(A, wb), b);
        double sum = 0;
        for (double x: diff) {
            sum += x*x;
        }
        System.out.println("The difference is: " + Math.sqrt(sum));
    }
}
