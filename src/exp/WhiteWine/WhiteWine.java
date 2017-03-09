package exp.WhiteWine;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying a breast cancer sample as being benign or malignant
 *
 * @author Jonathan Satria, adapted from AbaloneTest.java by Hannah Lau
 * @version 1.0
 */
class WhiteWineTest implements Runnable {
    private Thread t;


    private String results = "";

    private DecimalFormat df = new DecimalFormat("0.000");

    private static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
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
    private OptimizationAlgorithm oa;
    private BackPropagationNetwork network;
    private String oaName;
    private int trainingIterations;
    private ErrorMeasure measure;
    private Instance[] train_set;
    private Instance[] test_set;
    private ConcurrentHashMap<String, String> other_params;

    WhiteWineTest(
            OptimizationAlgorithm oa,
            BackPropagationNetwork network,
            String oaName,
            int trainingIterations,
            ErrorMeasure measure,
            Instance[] train_set,
            Instance[] test_set,
            ConcurrentHashMap<String, String> other_params
    ) {
        this.oa = oa;
        this.network = network;
        this.oaName = oaName;
        this.trainingIterations = trainingIterations;
        this.measure = measure;
        this.train_set = train_set;
        this.test_set = test_set;
        this.other_params = other_params;
    }
    public void run() {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(this.oa, this.network, this.oaName, this.trainingIterations);
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10,9);

                Instance optimalInstance = this.oa.getOptimal();
                this.network.setWeights(optimalInstance.getData());

                // Calculate Training Set Statistics //
                double predicted, actual;
                start = System.nanoTime();
                for(int j = 0; j < train_set.length; j++) {
                    this.network.setInputValues(train_set[j].getData());
                    this.network.run();

                    actual= Double.parseDouble(train_set[j].getLabel().toString());
                    predicted = Double.parseDouble(this.network.getOutputValues().toString());

                    double trash = Math.abs(Math.round(predicted * 10.0)/10.0 - actual) == 0.0 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10,9);
                //Train values
                results +=  "train," + this.oaName + "," + correct + "," + incorrect
                        + "," + df.format(correct/(correct+incorrect)*100)  //Percent correctly classified
                        + "," + df.format(trainingTime) //Training time seconds
                        + "," + df.format(testingTime)  //Test time seconds
                        + "," + this.trainingIterations + "\n"; //Iterations

                // Calculate Test Set Statistics //
                start = System.nanoTime();
                correct = 0; incorrect = 0;
                for(int j = 0; j < test_set.length; j++) {
                    this.network.setInputValues(test_set[j].getData());
                    this.network.run();

                    actual = Double.parseDouble(test_set[j].getLabel().toString());
                    predicted = Double.parseDouble(this.network.getOutputValues().toString());

                    double trash = Math.abs(Math.round(predicted * 10.0)/10.0 - actual) == 0.0 ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10,9);
                //Test values
                results += "test," + this.oaName + "," + correct + "," + incorrect
                        + "," + df.format(correct/(correct+incorrect)*100)  //Percent correctly classified
                        + "," + df.format(trainingTime) //Training time seconds
                        + "," + df.format(testingTime)  //Test time seconds
                        + "," + this.trainingIterations; //Iterations

        write_output_to_file(other_params.get("output_folder"), "final_results_neural_network.csv", results, true);
        System.out.println(results);
    }

    private void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += this.measure.value(output, example);
            }

            double test_error = 0;
            for(int j = 0; j < test_set.length; j++) {
                network.setInputValues(test_set[j].getData());
                network.run();

                Instance output = test_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                test_error += this.measure.value(output, example);
            }

            System.out.println(df.format(train_error)+","+df.format(test_error));
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

public class WhiteWine {
    private static int inputLayer = 11, hiddenLayer=10, outputLayer = 1;
    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];
    private static int rowCount = 4898;
    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, (int)(rowCount*0.75));
    private static Instance[] test_set = Arrays.copyOfRange(instances, (int)(rowCount*0.75), rowCount);
    private static DataSet set = new DataSet(train_set);
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static int[] trainingIterations = new int[] {10, 25, 50, 100, 200, 500 };
    private static String[] oaNames = {"RHC", "SA", "GA"};

    public static void main(String[] args) {
        ConcurrentHashMap<String, String> other_params = new ConcurrentHashMap<>();
        other_params.put("output_folder","Optimization_Results");

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);
        for(int i = 0; i < oa.length; i++) {
            for (int k = 0; k < trainingIterations.length; k++) {
                new WhiteWineTest(
                       oa[i],
                networks[i],
                oaNames[i],
                trainingIterations[k],
                measure,
                train_set,
                test_set,
                other_params
                ).start();
            }
        }
    }

    private static Instance[] initializeInstances() {
        int rowCount = WhiteWine.rowCount;
        int attributeCount = 11;
        double[][][] attributes = new double[rowCount][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("./src/exp/WhiteWine/winequality-white.csv")));

            //for each sample
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[attributeCount]; // 11 attributes
                attributes[i][1] = new double[1]; // classification

                // read features
                for(int j = 0; j < attributeCount; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            //We have 1-10 values to make decision upon. We are converting them into abagail format so that we can predict:
            instances[i].setLabel(new Instance(attributes[i][1][0] / 10.0));
        }

        return instances;
    }
}
