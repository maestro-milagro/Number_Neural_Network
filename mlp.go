package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"io"
	"math"
	"os"
	"strconv"
	"time"
)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

func NewNetwork(inputs, hiddens, outputs int, rate float64) *Network {
	net := Network{
		inputs:       inputs,
		hiddens:      hiddens,
		outputs:      outputs,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.outputs*net.hiddens, float64(net.hiddens)))

	return &net
}

func (net Network) Predict(inputData []float64) mat.Matrix {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)
	return finalOutputs
}

//
// Helper functions to allow easier use of Gonum
//

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// randomly generate a float64 array
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

func (net Network) Train(inputData, targetData []float64) {
	input := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, input)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	target := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(target, finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	net.outputWeights = add(net.outputWeights, scale(net.learningRate, dot(multiply(outputErrors, sigmoidPrime(finalOutputs)), hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = add(net.hiddenWeights, scale(net.learningRate, dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)), input.T()))).(*mat.Dense)
}

func (net *Network) save() error {
	hidden, err := os.Create("data/hweights.model")
	defer hidden.Close()
	if err != nil {
		return err
	}
	_, err = net.hiddenWeights.MarshalBinaryTo(hidden)
	if err != nil {
		return err
	}
	output, err := os.Create("data/outputs.model")
	defer output.Close()
	if err != nil {
		return err
	}
	_, err = net.outputWeights.MarshalBinaryTo(output)
	if err != nil {
		return err
	}

	return nil
}

func (net *Network) load() error {
	hidden, err := os.Open("data/hweights.model")
	defer hidden.Close()
	if err != nil {
		return err
	}
	net.hiddenWeights.Reset()
	_, err = net.hiddenWeights.UnmarshalBinaryFrom(hidden)
	if err != nil {
		return err
	}
	output, err := os.Open("data/outputs.model")
	defer output.Close()
	if err != nil {
		return err
	}
	net.outputWeights.Reset()
	_, err = net.outputWeights.UnmarshalBinaryFrom(output)
	if err != nil {
		return err
	}

	return nil
}
func (net *Network) mnistTrain() error {
	rand.Seed(uint64(time.Now().UTC().UnixNano()))
	t1 := time.Now()

	for epochs := 0; epochs < 5; epochs++ {
		tesFile, err := os.Open("mnist_dataset/mnist_train.csv")
		if err != nil {
			return err
		}
		r := csv.NewReader(bufio.NewReader(tesFile))
		for {
			record, err := r.Read()
			if err != nil {
				if err == io.EOF {
					break
				}
				tesFile.Close()
				return err
			}
			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 0.99 * 255.0) + 0.01
			}
			targets := make([]float64, net.outputs)
			for i := range targets {
				targets[i] = 0.01
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			net.Train(inputs, targets)
		}
		tesFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n\n", elapsed)
	return nil
}

func (net *Network) mnistPredict() error {
	rand.Seed(uint64(time.Now().UTC().UnixNano()))
	t1 := time.Now()

	file, err := os.Open("mnist_dataset/mnist_test.csv")
	if err != nil {
		return err
	}
	score := 0
	r := csv.NewReader(bufio.NewReader(file))
	for {
		record, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 0.99 * 255.0) + 0.01
		}
		outputs := net.Predict(inputs)

		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}
	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Printf("Score: %d\n", score)
	return nil
}
