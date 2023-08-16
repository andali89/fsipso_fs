package fs.pso;

import weka.attributeSelection.ASEvaluation;

public interface PSO {
	public void run();
	public double[] gbest();
	public double gbestFit();
	public double runtime();
	public double[] iterInfo();
	public void setSeed(long rnd);
	public void setEvaluator(ASEvaluation ASEval) ;
	public double[] iterNum();
	public double fitZero() ;
	public double[] iterMeanFit();	
	public double[] iterMeanNum() ;	
	public double[] iterMeanPbestFit();
	public double[] iterMeanPbestNum(); 
	public double getNum(double[] pos);
}
