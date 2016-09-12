package com.fatty.ml;

import java.util.Random;

/**
 * Created by fatty on 16/9/11.
 */
public class FastQPSolverDemo {
    public static void main(String[] args) {
        // Data preparation.
        int k = 50;
        int d = 20;
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
        int repeat = 50;

        // Solution A.
        double[] wa = null;
        double ta = System.nanoTime();
        for (int i=0; i<repeat; ++i)
            wa = UniformLLR.smoBased1dLLRSolver(Acap, bcap);
        ta = System.nanoTime()-ta;
        displayErr(A, b, wa);

        // Solution B. Compare with slow solver.
        double[] wb = null;
        double tb = System.nanoTime();
        for (int i=0; i<repeat; ++i)
            wb = UniformLLR.slowSolver(A, b);
        tb = System.nanoTime()-tb;
        displayErr(A, b, wb);

        // Solution C. Fast 2d solver.
        double[] wc = null;
        double tc = System.nanoTime();
        for (int i=0; i<repeat; ++i)
            wc = UniformLLR.smoBased2dLLRSolver(A, b);
        tc = System.nanoTime()-tc;
        displayErr(A, b, wc);

        System.out.printf("Time costed:\nA: %.3f, B: %.3f, C: %.3f\n", ta/1.0/repeat/1e6, tb/1.0/repeat/1e6, tc/1.0/repeat/1e6);
    }

    public static void displayErr(double[][] A, double[] b, double[] wb) {
        double[] diff = UniformLLR.vecSub(UniformLLR.matMultiply(A, wb), b);
        double sum = 0;
        for (double x: diff) {
            sum += x*x;
        }
        System.out.println("The difference is: " + Math.sqrt(sum));
    }
}
