package exp.FourPeaks;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.CountOnesEvaluationFunction;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * modified by Yeshwant Dattatreya
 * @version 1.0
 */

class Analyze_Optimization_Test_For_Time implements Runnable {

    private Thread t;

    private String problem;
    private String algorithm;
    private int iterations;
    private HashMap<String, Double> params;
    private int N;
    private int T;
    private ConcurrentHashMap<String, String> other_params;
    private int run;

    Analyze_Optimization_Test_For_Time(
            String problem,
            String algorithm,
            int iterations,
            HashMap<String, Double> params,
            int N,
            int T,
            ConcurrentHashMap<String, String> other_params,
            int run
    ) {
        this.problem = problem;
        this.algorithm = algorithm;
        this.iterations = iterations;
        this.params = params;
        this.N = N;
        this.T = T;
        this.other_params = other_params;
        this.run = run;
    }

    private void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void run() {
        try {
            EvaluationFunction ef = null;
            Distribution odd = null;
            NeighborFunction nf = null;
            MutationFunction mf = null;
            CrossoverFunction cf = null;
            Distribution df = null;
            int[] ranges;
            switch (this.problem) {
                case "count_ones":
                    ranges = new int[this.N];
                    Arrays.fill(ranges, 2);
                    ef = new CountOnesEvaluationFunction();
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new UniformCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges);
                    break;
                case "four_peaks":
                    ranges = new int[this.N];
                    Arrays.fill(ranges, 2);
                    ef = new FourPeaksEvaluationFunction(this.T);
                    odd = new DiscreteUniformDistribution(ranges);
                    nf = new DiscreteChangeOneNeighbor(ranges);
                    mf = new DiscreteChangeOneMutation(ranges);
                    cf = new SingleCrossOver();
                    df = new DiscreteDependencyTree(.1, ranges);
                    break;

            }
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            String results = "";
            double optimal_value = -1;
            boolean found = false;
            double start, elapsedTime = 0.0;
            double value = 0.0;
            int it = 0;

            switch (this.algorithm) {
                case "RHC":
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    start = System.currentTimeMillis();
                    do{
                        rhc.train();
                        value = ef.value(rhc.getOptimal());
                        it++;
                    } while (value != params.get("MAXIMUM_OPTIMA") && it < params.get("MAXIMUM_ITERATIONS"));
                    elapsedTime = System.currentTimeMillis() - start;

                    if (value == params.get("MAXIMUM_OPTIMA")){
                        found = true;
                    }
                    break;

                case "SA":
                    SimulatedAnnealing sa = new SimulatedAnnealing(
                            params.get("SA_initial_temperature"),
                            params.get("SA_cooling_factor"),
                            hcp
                    );
                    start = System.currentTimeMillis();
                    do{
                        sa.train();
                        value = ef.value(sa.getOptimal());
                        it++;
                    } while (value != params.get("MAXIMUM_OPTIMA") && it < params.get("MAXIMUM_ITERATIONS"));
                    elapsedTime = System.currentTimeMillis() - start;

                    if (value == params.get("MAXIMUM_OPTIMA")){
                        found = true;
                    }
                    break;

                case "GA":
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(
                            params.get("GA_population").intValue(),
                            params.get("GA_mate_number").intValue(),
                            params.get("GA_mutate_number").intValue(),
                            gap
                    );
                    start = System.currentTimeMillis();
                    do{
                        ga.train();
                        value = ef.value(ga.getOptimal());
                        it++;
                    } while (value != params.get("MAXIMUM_OPTIMA") && it < params.get("MAXIMUM_ITERATIONS"));
                    elapsedTime = System.currentTimeMillis() - start;

                    if (value == params.get("MAXIMUM_OPTIMA")){
                        found = true;
                    }
                    break;

                case "MIMIC":
                    MIMIC mimic = new MIMIC(
                            params.get("MIMIC_samples").intValue(),
                            params.get("MIMIC_to_keep").intValue(),
                            pop
                    );
                    start = System.currentTimeMillis();
                    do{
                        mimic.train();
                        value = ef.value(mimic.getOptimal());
                        it++;
                    } while (value != params.get("MAXIMUM_OPTIMA") && it < params.get("MAXIMUM_ITERATIONS"));
                    elapsedTime = System.currentTimeMillis() - start;

                    if (value == params.get("MAXIMUM_OPTIMA")){
                        found = true;
                    }
                    break;
            }
            results =
                    "Problem: " + this.problem + "\n" +
                    "Algorithm: " + this.algorithm + "\n" +
                    "Found: " + found +"\n" +
                    "Iterations: " + it + "\n" +
                    "Optimal Value: " + value + "\n" +
                    "Time Elapse: " + elapsedTime + "\n";
            String final_result = "";
            final_result =
                    this.problem + "," +
                            this.algorithm + "," +
                            this.N + "," +
                            this.iterations + "," +
                            this.run + "," +
                            optimal_value;
            write_output_to_file(this.other_params.get("output_folder"), "final_results.csv", final_result, true);
            String file_name =
                    this.problem + "_for_time_" + this.algorithm + "_N_" + this.N +
                            "_iter_" + this.iterations + "_run_" + this.run + ".csv";
            write_output_to_file(this.other_params.get("output_folder"), file_name, results, false);
            System.out.println(results);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void start () {
        if (t == null)
        {
            t = new Thread (this);
            t.start ();
        }
    }
}


public class OptimizationTestForTime {

    public static void main(String[] args) {

        ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
        other_params.put("output_folder","Optimization_Results");
        int num_runs = 10;

        //Count Ones Test
        HashMap<String, Double> count_one_test_params = new HashMap<>();
        count_one_test_params.put("SA_initial_temperature",10.);
        count_one_test_params.put("SA_cooling_factor",.95);
        count_one_test_params.put("GA_population",20.);
        count_one_test_params.put("GA_mate_number",20.);
        count_one_test_params.put("GA_mutate_number",5.);
        count_one_test_params.put("MIMIC_samples",50.);
        count_one_test_params.put("MIMIC_to_keep",10.);

        int[] N = {10,20};
        int[] iterations = {10,20,30};
        //"RHC", "SA", "GA",
        String[] algorithms = { "MIMIC"};
//        for (int i = 0; i < algorithms.length; i++) {
//            for (int j = 0; j < N.length; j++) {
//                //count_one_test_params.put("N",(double)N[j]);
//                for (int k = 0; k < iterations.length; k++) {
//                    for (int l = 0; l < num_runs; l++) {
//                        //other_params.remove("run");
//                        //other_params.put("run","" + l);
//                        new Analyze_Optimization_Test_For_Time(
//                                "count_ones",
//                                algorithms[i],
//                                iterations[k],
//                                count_one_test_params,
//                                N[j],
//                                0, //this doesn't apply to count ones problem, so simply pass a 0
//                                other_params,
//                                l
//                        );
//                    }
//                }
//            }
//        }

        //Four Peaks Test
        HashMap<String, Double> four_peaks_test_params = new HashMap<>();
        four_peaks_test_params.put("SA_initial_temperature",1E11);
        four_peaks_test_params.put("SA_cooling_factor",.95);
        four_peaks_test_params.put("GA_population",200.);
        four_peaks_test_params.put("GA_mate_number",100.);
        four_peaks_test_params.put("GA_mutate_number",10.);
        four_peaks_test_params.put("MIMIC_samples",200.);
        four_peaks_test_params.put("MIMIC_to_keep",20.);

        four_peaks_test_params.put("MAXIMUM_OPTIMA", 389.0);
        four_peaks_test_params.put("MAXIMUM_ITERATIONS", 50000.0);

        N = new int[] {200};
        iterations = new int[] {1000};
        for (int i = 0; i < algorithms.length; i++) {
            for (int j = 0; j < N.length; j++) {
                for (int k = 0; k < iterations.length; k++) {
                    for (int l = 0; l < num_runs; l++) {
                        //other_params.remove("run");
                        //other_params.put("run", "" + l);
                        new Analyze_Optimization_Test_For_Time(
                                "four_peaks",
                                algorithms[i],
                                iterations[k],
                                four_peaks_test_params,
                                N[j],
                                N[j]/10,
                                other_params,
                                l
                        ).start();
                    }
                }
            }
        }
    }
}

