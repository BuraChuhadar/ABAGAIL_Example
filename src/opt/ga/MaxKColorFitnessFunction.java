package opt.ga;

import java.util.Arrays;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A Max K Color evaluation function
 * @author kmandal
 * @version 1.0
 */
public class MaxKColorFitnessFunction implements EvaluationFunction {
    
    /**
     * 
     */
    private Vertex[] vertices;
    private int graphSize;
    
    public MaxKColorFitnessFunction(Vertex[] vertices) {
        this.vertices = vertices;
        this.graphSize = vertices.length;
        this.functionCallCount = 0;
    }

    private long functionCallCount;

    /**
     * @see opt.EvaluationFunction#getFunctionCallCount()
     */
    public long getFunctionCallCount()
    {
        return functionCallCount;
    }
    /**
     * @see opt.EvaluationFunction#setFunctionCallCount(opt.value)
     */
    public void setFunctionCallCount(long value)
    {
        functionCallCount = value;
    }

    private boolean conflict = false;

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     * Find how many iterations does it take to find if k-colors can be or can not be assigned to a given graph.
     */
    public double value(Instance d) {
        functionCallCount++;
        Vector data = d.getData();
        int n = data.size();
        double iterations = 0;
        conflict = false;
        //System.out.println("Sample color " + d.toString());
        for (int i = 0; i < n-1; i++) {
            int sampleColor = ((int) data.get(i));
            for(int j = 0; j < graphSize; j++){
              Vertex vertex = vertices[j];
              iterations ++;
             // System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
              if(vertex.getAadjacencyColorMatrix().contains(sampleColor)) {
            	  // if any of the adjacent vertices contains the color, the color can't be assigned to this vertex
            	  conflict = true;
            	  break;
              }
              //System.out.println("---------");
            }
        }
        return iterations;
    }

    
    public String foundConflict(){
    	return conflict ? "Failed to find Max-K Color combination !" : "Found Max-K Color Combination !";
    }
}
