/*
 * A Neural network implementation of XOR training set.
 * Created by: Shristi Pradhan (shristi.pr@gmail.com)
 */

package NNXOR;

import java.io.*;
import java.lang.Math;

/*
 * Class NeuralNetAlgo implements a neural network for XOR training set
 */
public class NeuralNetAlgo {
	
	private int numPattern;
	private int numInp;
	private int numInpBias;
	private int numHid;
	private int numWeightHidOut;
	private int flagBinary;
	
	private double [][]inp;
	private double []outp;
	
	private double[] outHid;
	private double[][]weightInHid;
	private double[] weightHidOut;
	private double[][] changeWeightInHid;
	private double[] changeWeightHidOut;
	
	private double rateLearn;
	private double momentum;
	
	/*
	 * Class Constructor
	 */
	public NeuralNetAlgo(int numPatternc,int numHidc,int flagBinaryc,double rateLearnc,double momentumc, double errLimit){
		numInp=2;
		numInpBias=numInp+1;
		numHid=numHidc;
		numWeightHidOut=numHid+1;
		flagBinary=flagBinaryc;
		numPattern=numPatternc;
		rateLearn=rateLearnc;
		momentum=momentumc;
		inp=new double[numPattern][numInpBias];
		outp=new double[numPattern];
		outHid=new double[numHid];
		weightInHid=new double[numInpBias][numHid];
		weightHidOut=new double[numWeightHidOut];
		changeWeightInHid=new double[numInpBias][numHid];
		changeWeightHidOut=new double[numWeightHidOut];
		
	}
	
	/*
	 * readData() checks if the input data is binary or bipolar and reads the input and output pattern accordingly.
	 * A bias term (input=1) is also added to the pattern to avoid the decision boundary to pass through the origin.
	 */
	public void readData(){
		
		if(flagBinary==1){
			System.out.println("Binary input");
		
		// Input and output patterns for binary data
		inp[0][0]=0;
		inp[0][1]=0;
		inp[0][2]=1;
		outp[0]=0;
		
		inp[1][0]=0;
		inp[1][1]=1;
		inp[1][2]=1;
		outp[1]=1;
		
		inp[2][0]=1;
		inp[2][1]=0;
		inp[2][2]=1;
		outp[2]=1;
		
		inp[3][0]=1;
		inp[3][1]=1;
		inp[3][2]=1;
		outp[3]=0;
		}
		
		else{
			System.out.println("Bipolar input");
			
			//input and output patterns for bipolar data
			inp[0][0]=-1;
			inp[0][1]=-1;
			inp[0][2]=1;
			outp[0]=-1;
			
			inp[1][0]=-1;
			inp[1][1]=1;
			inp[1][2]=1;
			outp[1]=1;
			
			inp[2][0]=1;
			inp[2][1]=-1;
			inp[2][2]=1;
			outp[2]=1;
			
			inp[3][0]=1;
			inp[3][1]=1;
			inp[3][2]=1;
			outp[3]=-1;
		}
	
	}
	
	/*
	 * initializeWeight() initializes weights of the network to random values in the range [-0.5,+0.5]
	 */
	public void initializeWeight(){
		System.out.println("Initializing weights");
		
		for(int i=0;i<numWeightHidOut;i++){
			weightHidOut[i]=(Math.random()-0.5);
		}
		for(int i=0;i<numHid;i++){
			for(int j=0;j<numInpBias;j++){
				weightInHid[j][i]=(Math.random()-0.5);
			}
		}
	}
	
	/*
	 * Feedforward: outputFor() computes the output of the network
	 */
	public double outputFor(int patternNum){
		for(int i=0;i<numHid;i++){
			outHid[i]=0.0;
			for(int j=0;j<numInpBias;j++){
				outHid[i]=outHid[i]+(inp[patternNum][j]*weightInHid[j][i]);
			}
			outHid[i]=sigmoid(outHid[i]);
		}
		
		double nnOutput=0.0;
		for(int i=0;i<numWeightHidOut;i++){
			if(i<numHid){
				nnOutput+=outHid[i]*weightHidOut[i];
			}
			else{
				nnOutput=nnOutput+weightHidOut[i];
			}
		}
		return sigmoid(nnOutput);
		
	}
	
	/*
	 * Backpropagation of error: train() trains the network. Weights of the neural network are updated.
	 */
	public void train(int patternNum,double networkOut){
		
		if(flagBinary==1){
			
			// Weight updates of hidden to output layer
			for(int i=0;i<numWeightHidOut;i++){
				if(i<numHid){
					
					weightHidOut[i]+=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*networkOut*(1-networkOut)*outHid[i];
					changeWeightHidOut[i]=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*networkOut*(1-networkOut)*outHid[i];
				
				}
				
				// Bias term
				else{
				weightHidOut[i]+=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*networkOut*(1-networkOut);
				changeWeightHidOut[i]=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*networkOut*(1-networkOut);
				}
			}
			
			// Weight updates of input to hidden layer
			double weightOutErr;
			for(int i=0;i<numHid;i++){
				weightOutErr=(outp[patternNum]-networkOut)*networkOut*(1-networkOut)*weightHidOut[i];
				for(int j=0;j<numInpBias;j++){
				weightInHid[j][i]+=momentum*changeWeightInHid[j][i]+rateLearn*weightOutErr*outHid[i]*(1-outHid[i])*inp[patternNum][j];
				changeWeightInHid[j][i]=momentum*changeWeightInHid[j][i]+rateLearn*weightOutErr*outHid[i]*(1-outHid[i])*inp[patternNum][j];
				}
			}	
		}
		
		else{
			// Weight updates of hidden to output layer
			for(int i=0;i<numWeightHidOut;i++){
				if(i<numHid){
					weightHidOut[i]+=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*0.5*(1-networkOut*networkOut)*outHid[i];
					changeWeightHidOut[i]=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*0.5*(1-networkOut*networkOut)*outHid[i];
				}
				else{
					// Bias term
					weightHidOut[i]+=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*0.5*(1-networkOut*networkOut);
					changeWeightHidOut[i]=momentum*changeWeightHidOut[i]+rateLearn*(outp[patternNum]-networkOut)*0.5*(1-networkOut*networkOut);
					}	
			}
			
			// Weight updates of input to hidden layer
			double weightOutErr;
			for(int i=0;i<numHid;i++){
				weightOutErr=(outp[patternNum]-networkOut)*0.5*(1-networkOut*networkOut)*weightHidOut[i];
				for(int j=0;j<numInpBias;j++){
					weightInHid[j][i]+=momentum*changeWeightInHid[j][i]+rateLearn*weightOutErr*0.5*(1-outHid[i]*outHid[i])*inp[patternNum][j];
					changeWeightInHid[j][i]=momentum*changeWeightInHid[j][i]+rateLearn*weightOutErr*0.5*(1-outHid[i]*outHid[i])*inp[patternNum][j];
				}
			}
		}
		
	}
	
	/*
	 * computeTotErr() returns the error for the current pattern
	 */
	public double computeTotErr(int patternNum,double networkOut){
		
		double err=outp[patternNum]-networkOut;
		return 0.5*err*err;
	}
	
	/*
	 *computeAvgErr() return the average error over all the patterns 
	 */
	public double computeAvgErr(){
		double err=0;
		for(int i=0;i<numPattern;i++){
			err+=Math.abs(outp[i]-outputFor(i));
		}
		return 0.25*err;
	}
	
	/*
	 * sigmoid() returns the sigmoidal output depending on the binary of bipolar input pattern
	 */
	public double sigmoid(double x){
		if(flagBinary==1){
			return 1/(1+Math.exp(-x));
			}
		else{
			return (1-Math.exp(-x))/(1+Math.exp(-x));
		}
	}
	
	/*
	 * saveWeights() saves the values of weights to a .txt file
	 */
	public void saveWeight(){
		try{
			FileWriter fstreamWeightValue=new FileWriter("WeightValue.txt");
			BufferedWriter outWeightValue=new BufferedWriter(fstreamWeightValue);
			outWeightValue.write("Weights from Hidden to Output Layer:");
			outWeightValue.newLine();
			
			for(int i=0;i<numWeightHidOut;i++){
				outWeightValue.write(weightHidOut[i]+"");
				outWeightValue.newLine();
			}
			outWeightValue.newLine();
			outWeightValue.write("Weights from Input to Hidden Layer:");
			outWeightValue.newLine();
			
			for(int i=0;i<numHid;i++){
				outWeightValue.newLine();
				outWeightValue.write("Input to Hidden Layer");
				outWeightValue.newLine();
				
				for(int j=0;j<numInpBias;j++){
					outWeightValue.write(weightHidOut[i]+"");
					outWeightValue.newLine();
				}
			}
			
			outWeightValue.close();
		}
		catch(Exception e){
			System.err.println("ERROR:"+e.getMessage());
		}
	}

}
