package exp.CountOnes;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.CountOnesEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.Arrays;

/**
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest {
    /** The n value */
    private static final int N = 1000;

    public static void main(String[] args) {
        int max_iterations=20000;
        int iterations = 0;
        int optimum = 0;
        int[] N = {20,40,60,80,100, 150, 200, 300, 500, 1000};

        for (int i=0; i < N.length; i++) {
            int[] ranges = new int[N[i]];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            System.out.println("Testing for N: " + N[i]);
            optimum = N[i];
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 1);
            do {
                fit.train();
                iterations++;
            } while (ef.value(rhc.getOptimal()) != optimum && iterations < max_iterations);
            System.out.println(iterations + " " + ef.value(rhc.getOptimal()));

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            iterations = 0;
            fit = new FixedIterationTrainer(sa, 1);
            do {
                fit.train();
                iterations++;
            } while (ef.value(sa.getOptimal()) != optimum && iterations < max_iterations);
            System.out.println(iterations + " " + ef.value(sa.getOptimal()));

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 1, gap);
            iterations = 0;
            fit = new FixedIterationTrainer(ga, 1);
            do {
                fit.train();
                iterations++;
            } while (ef.value(ga.getOptimal()) != optimum && iterations < max_iterations);
            System.out.println(iterations + " " + ef.value(ga.getOptimal()));

            MIMIC mimic = new MIMIC(50, 10, pop);
            iterations = 0;
            fit = new FixedIterationTrainer(mimic, 1);
            do {
                fit.train();
                iterations++;
                if (ef.value(mimic.getOptimal()) == optimum) {
                    System.out.println("found!" + iterations + " " + ef.value(mimic.getOptimal()));
                    break;
                }
            } while (ef.value(mimic.getOptimal()) != optimum && iterations < 1000);
            System.out.println(iterations + " " + ef.value(mimic.getOptimal()));
        }
    }
}