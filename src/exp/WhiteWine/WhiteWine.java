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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying a breast cancer sample as being benign or malignant
 *
 * @author Jonathan Satria, adapted from AbaloneTest.java by Hannah Lau
 * @version 1.0
 */
public class WhiteWine {
    private static int rowCount = 4898;
    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, (int)(rowCount*0.75));
    private static Instance[] test_set = Arrays.copyOfRange(instances, (int)(rowCount*0.75), rowCount);

    private static DataSet set = new DataSet(train_set);

    private static int inputLayer = 11, hiddenLayer=6, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            // Calculate Training Set Statistics //
            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < train_set.length; j++) {
                networks[i].setInputValues(train_set[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(train_set[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = (Math.abs(predicted - actual)*10.0) < 1.0 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nTrain Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            // Calculate Test Set Statistics //
            start = System.nanoTime();
            correct = 0; incorrect = 0;
            for(int j = 0; j < test_set.length; j++) {
                networks[i].setInputValues(test_set[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(test_set[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = (Math.abs(predicted - actual)*10.0) < 1.0 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nTest Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }

            double test_error = 0;
            for(int j = 0; j < test_set.length; j++) {
                network.setInputValues(test_set[j].getData());
                network.run();

                Instance output = test_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                test_error += measure.value(output, example);
            }

            System.out.println(df.format(train_error)+","+df.format(test_error));
        }
    }

    private static Instance[] initializeInstances() {
        int rowCount = 4898;
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
