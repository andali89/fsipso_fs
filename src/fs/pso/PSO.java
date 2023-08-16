/** 
 * Copyright (c) 2023, An-Da Li. All rights reserved. 
 * Please read LICENCE for license terms.
 * Coded by An-Da Li
 * Email: andali1989@163.com
 *
 * Li, A.-D., Xue, B., & Zhang, M. (2021). A Forward Search Inspired Particle Swarm Optimization Algorithm 
 * for Feature Selection in Classification. IEEE Congress on Evolutionary Computation, CEC 2021, Kraków, 
 * Poland, June 28 - July 1, 2021, 786–793. https://doi.org/10.1109/CEC45853.2021.9504949
 *
 */

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
